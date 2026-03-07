"""
Ontology pipeline node: build an rdflib OWL graph from the metadata report.

Mapping strategy
----------------
Table           -> owl:Class
  PK column     -> owl:DatatypeProperty + owl:FunctionalProperty + owl:InverseFunctionalProperty
  NOT NULL col  -> owl:DatatypeProperty + rdfs:subClassOf owl:Restriction (minCardinality 1)
  nullable col  -> owl:DatatypeProperty
FK candidate    -> owl:ObjectProperty (domain = left table, range = right table)
Explicit FK     -> owl:ObjectProperty
Cardinality 1:1 -> owl:FunctionalProperty + owl:InverseFunctionalProperty on the object property
Cardinality 1:N -> owl:FunctionalProperty on the object property
FD              -> rdfs:comment annotation on the owning class
Statistics      -> rdfs:comment annotations on datatype properties (if include_statistics=True)
"""
from __future__ import annotations

import logging
import re
from typing import Dict, Set, Tuple

from rdflib import BNode, Graph, Literal, Namespace, RDF, RDFS, OWL, URIRef, XSD

from ..state import OntologyState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# XSD type mapping
# ---------------------------------------------------------------------------

_TYPE_MAP: Dict[str, URIRef] = {
    "varchar":            XSD.string,
    "character varying":  XSD.string,
    "char":               XSD.string,
    "text":               XSD.string,
    "string":             XSD.string,
    "nvarchar":           XSD.string,
    "nchar":              XSD.string,
    "clob":               XSD.string,
    "json":               XSD.string,
    "jsonb":              XSD.string,
    "uuid":               XSD.string,
    "int":                XSD.integer,
    "integer":            XSD.integer,
    "int4":               XSD.integer,
    "int8":               XSD.integer,
    "bigint":             XSD.integer,
    "smallint":           XSD.integer,
    "tinyint":            XSD.integer,
    "serial":             XSD.integer,
    "bigserial":          XSD.integer,
    "number":             XSD.decimal,
    "numeric":            XSD.decimal,
    "decimal":            XSD.decimal,
    "float":              XSD.double,
    "float4":             XSD.double,
    "float8":             XSD.double,
    "double":             XSD.double,
    "double precision":   XSD.double,
    "real":               XSD.float,
    "boolean":            XSD.boolean,
    "bool":               XSD.boolean,
    "date":               XSD.date,
    "timestamp":          XSD.dateTime,
    "datetime":           XSD.dateTime,
    "timestamp with time zone":    XSD.dateTime,
    "timestamp without time zone": XSD.dateTime,
    "time":               XSD.time,
    "interval":           XSD.duration,
    "bytea":              XSD.hexBinary,
    "blob":               XSD.hexBinary,
}


def _xsd_type(db_type: str) -> URIRef:
    """Map a DB data type string to the nearest XSD type."""
    dt = db_type.lower().split("(")[0].strip()
    return _TYPE_MAP.get(dt, XSD.string)


def _safe(name: str) -> str:
    """Convert a table/column name to a safe URI fragment."""
    s = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if s and s[0].isdigit():
        s = "_" + s
    return s or "unknown"


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def build_node(state: OntologyState) -> OntologyState:
    config = state["config"]
    report = state["report"]

    base_uri = config.base_uri.rstrip("/") + "/" + _safe(config.ontology_name) + "/"
    ns       = Namespace(base_uri)
    onto_uri = URIRef(base_uri.rstrip("/"))

    g = Graph()
    g.bind("",    ns)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("xsd",  XSD)

    # Ontology header
    g.add((onto_uri, RDF.type,    OWL.Ontology))
    g.add((onto_uri, RDFS.label,  Literal(config.ontology_name)))
    g.add((onto_uri, RDFS.comment, Literal(
        f"Generated from database: {report.get('schema','unknown')} "
        f"({report.get('database_type','unknown')})"
    )))

    tables:      Dict = report.get("tables") or {}
    class_map:   Dict = {}
    property_map: Dict = {}
    added_obj_props: Set[URIRef] = set()

    # ------------------------------------------------------------------
    # 1. OWL Classes from tables + DatatypeProperties from columns
    # ------------------------------------------------------------------
    for table_name, table_meta in tables.items():
        cls_uri = ns[_safe(table_name)]
        g.add((cls_uri, RDF.type,    OWL.Class))
        g.add((cls_uri, RDFS.label,  Literal(table_name)))

        if isinstance(table_meta, dict):
            row_count = table_meta.get("row_count")
            if row_count is not None:
                g.add((cls_uri, RDFS.comment, Literal(f"row_count={row_count}")))

        class_map[table_name] = cls_uri
        logger.debug("  Class: %s", table_name)

        if not isinstance(table_meta, dict):
            continue

        for col in table_meta.get("columns") or []:
            if not isinstance(col, dict):
                continue

            col_name  = col.get("name", "")
            prop_uri  = ns[f"{_safe(table_name)}_{_safe(col_name)}"]
            xsd_range = _xsd_type(col.get("data_type", ""))

            g.add((prop_uri, RDF.type,   OWL.DatatypeProperty))
            g.add((prop_uri, RDFS.label, Literal(col_name)))
            g.add((prop_uri, RDFS.domain, cls_uri))
            g.add((prop_uri, RDFS.range,  xsd_range))

            is_pk = col.get("is_primary_key", False)
            if is_pk:
                g.add((prop_uri, RDF.type, OWL.FunctionalProperty))
                g.add((prop_uri, RDF.type, OWL.InverseFunctionalProperty))

            if not col.get("nullable", True):
                restriction = BNode()
                g.add((restriction, RDF.type,           OWL.Restriction))
                g.add((restriction, OWL.onProperty,     prop_uri))
                g.add((restriction, OWL.minCardinality,
                       Literal(1, datatype=XSD.nonNegativeInteger)))
                g.add((cls_uri, RDFS.subClassOf, restriction))

            if config.include_statistics:
                parts = []
                for key in ("unique_count", "null_count", "min_value", "max_value", "avg_value"):
                    val = col.get(key)
                    if val is not None:
                        parts.append(f"{key}={val}")
                if parts:
                    g.add((prop_uri, RDFS.comment, Literal(", ".join(parts))))

            property_map[(table_name, col_name)] = prop_uri

    # ------------------------------------------------------------------
    # 2. ObjectProperties from explicit FK definitions
    # ------------------------------------------------------------------
    for table_name, table_meta in tables.items():
        if not isinstance(table_meta, dict):
            continue
        for fk in table_meta.get("foreign_keys") or []:
            if not isinstance(fk, dict):
                continue
            ref_table = (
                fk.get("ref_table")
                or fk.get("referenced_table")
                or fk.get("foreign_table")
                or ""
            )
            if not ref_table or ref_table not in class_map:
                continue
            prop_uri = ns[f"{_safe(table_name)}_fk_{_safe(ref_table)}"]
            if prop_uri in added_obj_props:
                continue
            g.add((prop_uri, RDF.type,    OWL.ObjectProperty))
            g.add((prop_uri, RDFS.label,  Literal(f"{table_name} → {ref_table} (FK)")))
            g.add((prop_uri, RDFS.domain, class_map[table_name]))
            g.add((prop_uri, RDFS.range,  class_map[ref_table]))
            added_obj_props.add(prop_uri)

    # ------------------------------------------------------------------
    # 3. ObjectProperties from FK candidates (inclusion dependencies)
    # ------------------------------------------------------------------
    for fk in report.get("fk_candidates") or []:
        left_t  = fk.get("left_table",  "")
        right_t = fk.get("right_table", "")
        if left_t not in class_map or right_t not in class_map:
            continue
        prop_uri = ns[f"{_safe(left_t)}_references_{_safe(right_t)}"]
        if prop_uri in added_obj_props:
            continue
        g.add((prop_uri, RDF.type,    OWL.ObjectProperty))
        g.add((prop_uri, RDFS.label,  Literal(f"{left_t} references {right_t}")))
        g.add((prop_uri, RDFS.domain, class_map[left_t]))
        g.add((prop_uri, RDFS.range,  class_map[right_t]))
        g.add((prop_uri, RDFS.comment,
               Literal(f"FK candidate — coverage={fk.get('coverage', 0):.3f}")))
        added_obj_props.add(prop_uri)

    # ------------------------------------------------------------------
    # 4. ObjectProperties + cardinality from cardinality relationships
    # ------------------------------------------------------------------
    for cr in report.get("cardinality_relationships") or []:
        left_t   = cr.get("left_table",  "")
        right_t  = cr.get("right_table", "")
        rel_type = cr.get("type", "M:N")
        if left_t not in class_map or right_t not in class_map:
            continue
        prop_uri = ns[f"{_safe(left_t)}_relates_{_safe(right_t)}"]
        if prop_uri not in added_obj_props:
            g.add((prop_uri, RDF.type,    OWL.ObjectProperty))
            g.add((prop_uri, RDFS.label,  Literal(f"{left_t} ↔ {right_t}")))
            g.add((prop_uri, RDFS.domain, class_map[left_t]))
            g.add((prop_uri, RDFS.range,  class_map[right_t]))
            added_obj_props.add(prop_uri)
        g.add((prop_uri, RDFS.comment, Literal(f"cardinality={rel_type}")))
        if rel_type == "1:1":
            g.add((prop_uri, RDF.type, OWL.FunctionalProperty))
            g.add((prop_uri, RDF.type, OWL.InverseFunctionalProperty))
        elif rel_type in ("1:N", "N:1"):
            g.add((prop_uri, RDF.type, OWL.FunctionalProperty))

    # ------------------------------------------------------------------
    # 5. Annotate classes with FD summaries
    # ------------------------------------------------------------------
    for fd in report.get("functional_dependencies") or []:
        tbl  = fd.get("table", "")
        if tbl not in class_map:
            continue
        det  = ", ".join(fd.get("determinant") or [])
        dep  = ", ".join(fd.get("dependent")   or [])
        conf = fd.get("confidence", 0)
        g.add((class_map[tbl], RDFS.comment,
               Literal(f"FD: [{det}] → [{dep}]  conf={conf:.3f}")))

    state["ontology_graph"]  = g
    state["class_map"]       = class_map
    state["property_map"]    = property_map
    state["class_count"]     = len(class_map)
    state["property_count"]  = len(property_map) + len(added_obj_props)
    state["triple_count"]    = len(g)
    state["phase"]           = "built"

    logger.info(
        "Ontology built: %d classes, %d datatype props, %d object props, %d triples",
        len(class_map), len(property_map), len(added_obj_props), len(g),
    )
    return state
