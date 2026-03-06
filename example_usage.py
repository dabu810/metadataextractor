"""
Example usage of the Metadata Extraction Agent.

Demonstrates all supported databases.  Set the connection details in the
config and call agent.run() to extract full schema metadata.
"""
import json
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from metadata_agent import AgentConfig, DBConfig, DBType, MetadataExtractionAgent


# ===========================================================================
# PostgreSQL example
# ===========================================================================
def run_postgres():
    config = AgentConfig(
        db_config=DBConfig(
            db_type=DBType.POSTGRES,
            host="localhost",
            port=5432,
            database="mydb",
            schema="public",
            username=os.environ["PG_USER"],
            password=os.environ["PG_PASSWORD"],
        ),
        target_tables=None,         # None = scan all tables
        sample_size=50_000,
        fd_threshold=1.0,           # exact FDs only
        id_threshold=0.95,
        output_path="reports/postgres_metadata.json",
    )
    agent = MetadataExtractionAgent(config)
    report = agent.run()

    print("\n=== Summary ===")
    print(json.dumps(report["summary"], indent=2))

    # Ask the LLM a question about the results
    answer = agent.ask(
        "Which tables have the most null values in their columns, "
        "and what are the strongest functional dependencies?"
    )
    print("\n=== LLM Insights ===")
    print(answer)
    return report


# ===========================================================================
# BigQuery example
# ===========================================================================
def run_bigquery():
    config = AgentConfig(
        db_config=DBConfig(
            db_type=DBType.BIGQUERY,
            project="my-gcp-project",
            schema="my_dataset",
            credentials_path="/path/to/service_account.json",
        ),
        sample_size=10_000,
        fd_threshold=0.99,          # allow 1% violations (approximate FDs)
        id_threshold=0.90,
        output_path="reports/bigquery_metadata.json",
    )
    agent = MetadataExtractionAgent(config)
    report = agent.run()
    return report


# ===========================================================================
# Oracle example
# ===========================================================================
def run_oracle():
    config = AgentConfig(
        db_config=DBConfig(
            db_type=DBType.ORACLE,
            host="oracle-host.example.com",
            port=1521,
            database="ORCL",           # service_name
            schema="HR",
            username=os.environ["ORA_USER"],
            password=os.environ["ORA_PASSWORD"],
        ),
        target_tables=["EMPLOYEES", "DEPARTMENTS", "JOBS"],
        sample_size=20_000,
        output_path="reports/oracle_metadata.json",
    )
    agent = MetadataExtractionAgent(config)
    return agent.run()


# ===========================================================================
# SQL Server example
# ===========================================================================
def run_sqlserver():
    config = AgentConfig(
        db_config=DBConfig(
            db_type=DBType.SQLSERVER,
            host="sqlserver.example.com",
            port=1433,
            database="AdventureWorks",
            schema="Sales",
            username=os.environ["MSSQL_USER"],
            password=os.environ["MSSQL_PASSWORD"],
            extra={"driver": "ODBC Driver 18 for SQL Server"},
        ),
        sample_size=30_000,
        output_path="reports/sqlserver_metadata.json",
    )
    agent = MetadataExtractionAgent(config)
    return agent.run()


# ===========================================================================
# Teradata example
# ===========================================================================
def run_teradata():
    config = AgentConfig(
        db_config=DBConfig(
            db_type=DBType.TERADATA,
            host="teradata.example.com",
            database="MY_DB",
            schema="MY_DB",
            username=os.environ["TD_USER"],
            password=os.environ["TD_PASSWORD"],
            extra={"logmech": "TD2"},
        ),
        sample_size=10_000,
        output_path="reports/teradata_metadata.json",
    )
    agent = MetadataExtractionAgent(config)
    return agent.run()


# ===========================================================================
# Redshift example
# ===========================================================================
def run_redshift():
    config = AgentConfig(
        db_config=DBConfig(
            db_type=DBType.REDSHIFT,
            host="my-cluster.abc123.us-east-1.redshift.amazonaws.com",
            port=5439,
            database="dev",
            schema="public",
            username=os.environ["RS_USER"],
            password=os.environ["RS_PASSWORD"],
        ),
        sample_size=50_000,
        output_path="reports/redshift_metadata.json",
    )
    agent = MetadataExtractionAgent(config)
    return agent.run()


# ===========================================================================
# Delta Lake / Databricks example
# ===========================================================================
def run_delta_databricks():
    config = AgentConfig(
        db_config=DBConfig(
            db_type=DBType.DELTA_LAKE,
            host="adb-123456789.azuredatabricks.net",
            database="my_catalog",
            schema="my_schema",
            password=os.environ["DATABRICKS_TOKEN"],   # personal access token
            extra={"http_path": "/sql/1.0/warehouses/abc123"},
        ),
        sample_size=10_000,
        output_path="reports/delta_metadata.json",
    )
    agent = MetadataExtractionAgent(config)
    return agent.run()


def run_delta_local():
    """PySpark + local Delta tables."""
    config = AgentConfig(
        db_config=DBConfig(
            db_type=DBType.DELTA_LAKE,
            schema="default",
            spark_master="local[*]",
        ),
        sample_size=5_000,
        output_path="reports/delta_local_metadata.json",
    )
    agent = MetadataExtractionAgent(config)
    return agent.run()


# ===========================================================================
# Streaming progress example
# ===========================================================================
def run_with_streaming(config: AgentConfig):
    """Show node-by-node progress as the pipeline executes."""
    agent = MetadataExtractionAgent(config)
    print("Streaming pipeline execution …\n")
    for node_name, _ in agent.stream_run():
        print(f"  ✓ Node '{node_name}' completed")
    print("\nFinal report:")
    print(json.dumps(agent._report.get("summary", {}), indent=2))


# ===========================================================================
if __name__ == "__main__":
    # Change this to whichever DB you want to test
    report = run_postgres()
    print(f"\nExtracted metadata for {report['summary']['total_tables']} tables.")
    print(f"Functional dependencies : {report['summary']['total_functional_dependencies']}")
    print(f"Inclusion dependencies  : {report['summary']['total_inclusion_dependencies']}")
    print(f"FK candidates           : {report['summary']['total_fk_candidates']}")
    print(f"Cardinality links       : {report['summary']['total_cardinality_relationships']}")
