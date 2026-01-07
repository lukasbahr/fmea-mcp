"""
Neo4J Service for FMEA Graph Management
"""

import pandas as pd
from typing import Optional
from neo4j import GraphDatabase
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph


class Neo4JService:
    """Neo4J Service for FMEA graph operations with vector embeddings support."""

    # Cypher queries
    MERGE_NODE_QUERY = "MERGE ({nodeRef}:{node} {properties})"
    MERGE_RELATION_QUERY = "MERGE ({nodeRef1})-[:{relation}]->({nodeRef2})"
    MATCH_QUERY = "MATCH ({nodeRef}:{node} {properties})"

    TRAVERSE_QUERY = """
    MATCH (fm:FailureMeasure)<-[:isImprovedByFailureMeasure]-(fc:FailureCause)<-[:isDueToFailureCause]-(fd:FailureMode)-[:occursAtProcessStep]->(ps:ProcessStep)
    WITH fm, fc, fd, ps
    MATCH (fd)-[:resultsInFailureEffect]->(fe:FailureEffect)
    WHERE ID(fd)={id}
    RETURN fm, fc, fe, fd, ps, ID(fm), ID(fc), ID(fe), ID(fd), ID(ps);
    """

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        enable_embeddings: bool = False,
        openai_api_key: Optional[str] = None,
    ):
        """
        Initialize Neo4J connection.

        Args:
            uri: Neo4J URI
            username: Neo4J username
            password: Neo4J password
            database: Neo4J database name
            enable_embeddings: Whether to enable vector embeddings
            openai_api_key: OpenAI API key for embeddings (required if enable_embeddings=True)
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        self.enable_embeddings = enable_embeddings
        self.vector_store = None
        self.graph = None

        if enable_embeddings:
            if not openai_api_key:
                raise ValueError(
                    "OpenAI API key is required when embeddings are enabled"
                )

            try:
                # Initialize embeddings
                self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

                # Initialize vector store
                self.vector_store = Neo4jVector.from_existing_graph(
                    embedding=self.embeddings,
                    url=uri,
                    username=username,
                    password=password,
                    database=database,
                    index_name="fmea_embeddings",
                    node_label="Chunk",
                    text_node_properties=["text"],
                    embedding_node_property="embedding",
                )

                # Initialize graph
                self.graph = Neo4jGraph(
                    url=uri, username=username, password=password, database=database
                )
            except Exception as e:
                print(f"Warning: Could not initialize embeddings: {e}")
                self.enable_embeddings = False

    def close(self):
        """Close the Neo4J connection."""
        if self.driver:
            self.driver.close()

    @staticmethod
    def format_properties(properties: dict) -> str:
        """
        Formats a dictionary of properties into a string representation.

        Args:
            properties (dict): A dictionary of properties to format.

        Returns:
            str: A string representation of the formatted properties.
        """
        if not properties:
            return "{}"

        properties_str = "{"
        for key, value in properties.items():
            if key in ["S", "O", "D", "RPN"]:
                properties_str += f"{key}: {value},"
            else:
                # Escape quotes in string values
                escaped_value = str(value).replace('"', '\\"')
                properties_str += f'{key}: "{escaped_value}",'

        properties_str = properties_str.rstrip(",") + "}"
        return properties_str

    def query(self, cypher: str) -> list:
        """
        Execute a Cypher query.

        Args:
            cypher: The Cypher query to execute

        Returns:
            list: Query results
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher)
            return [record.data() for record in result]

    def create_fmea_graph(self, csv_file_path: str) -> dict:
        """
        Create the FMEA graph from a CSV file.

        Args:
            csv_file_path: Path to the CSV file containing FMEA data

        Returns:
            dict: Status and statistics of the graph creation

        Expected CSV columns:
        - ProcessStep
        - FailureMode
        - FailureEffect
        - FailureCause
        - FailureMeasure
        - DetectionMeasure
        - S (Severity)
        - O (Occurrence)
        - D (Detection)
        - RPN (Risk Priority Number)
        """
        try:
            # Read CSV file
            df = pd.read_csv(csv_file_path, delimiter=";", encoding="utf-8")

            # Validate required columns
            required_columns = [
                "ProcessStep",
                "FailureMode",
                "FailureEffect",
                "FailureCause",
                "FailureMeasure",
                "DetectionMeasure",
                "S",
                "O",
                "D",
                "RPN",
            ]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return {
                    "success": False,
                    "error": f"Missing required columns: {', '.join(missing_columns)}",
                }

            nodes_created = 0
            relations_created = 0

            # Create nodes and relations for each row
            for idx, row in df.iterrows():
                nodes = [
                    self.MERGE_NODE_QUERY.format(
                        nodeRef="FailureMode",
                        node="FailureMode",
                        properties=self.format_properties(
                            {
                                "FailureMode": row["FailureMode"],
                                "RPN": int(row["RPN"]),
                            }
                        ),
                    ),
                    self.MERGE_NODE_QUERY.format(
                        nodeRef="ProcessStep",
                        node="ProcessStep",
                        properties=self.format_properties(
                            {"ProcessStep": row["ProcessStep"]}
                        ),
                    ),
                    self.MERGE_NODE_QUERY.format(
                        nodeRef="FailureEffect",
                        node="FailureEffect",
                        properties=self.format_properties(
                            {
                                "FailureEffect": row["FailureEffect"],
                                "S": int(row["S"]),
                            }
                        ),
                    ),
                    self.MERGE_NODE_QUERY.format(
                        nodeRef="FailureCause",
                        node="FailureCause",
                        properties=self.format_properties(
                            {
                                "FailureCause": row["FailureCause"],
                                "O": int(row["O"]),
                            }
                        ),
                    ),
                    self.MERGE_NODE_QUERY.format(
                        nodeRef="FailureMeasure",
                        node="FailureMeasure",
                        properties=self.format_properties(
                            {
                                "FailureMeasure": row["FailureMeasure"],
                                "DetectionMeasure": row["DetectionMeasure"],
                                "D": int(row["D"]),
                            }
                        ),
                    ),
                ]

                relations = [
                    self.MERGE_RELATION_QUERY.format(
                        nodeRef1="FailureMode",
                        relation="occursAtProcessStep",
                        nodeRef2="ProcessStep",
                    ),
                    self.MERGE_RELATION_QUERY.format(
                        nodeRef1="FailureMode",
                        relation="resultsInFailureEffect",
                        nodeRef2="FailureEffect",
                    ),
                    self.MERGE_RELATION_QUERY.format(
                        nodeRef1="FailureMode",
                        relation="isDueToFailureCause",
                        nodeRef2="FailureCause",
                    ),
                    self.MERGE_RELATION_QUERY.format(
                        nodeRef1="FailureCause",
                        relation="isImprovedByFailureMeasure",
                        nodeRef2="FailureMeasure",
                    ),
                ]

                query = "\n".join(nodes + relations)

                try:
                    self.query(query)
                    nodes_created += len(nodes)
                    relations_created += len(relations)
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Error processing row {idx}: {str(e)}",
                    }

            # Create vector embeddings if enabled
            embeddings_created = 0
            if self.enable_embeddings:
                try:
                    embeddings_result = self.create_vector_embeddings()
                    if embeddings_result["success"]:
                        embeddings_created = embeddings_result.get(
                            "embeddings_created", 0
                        )
                except Exception as e:
                    print(f"Warning: Could not create embeddings: {e}")

            return {
                "success": True,
                "rows_processed": len(df),
                "nodes_created": nodes_created,
                "relations_created": relations_created,
                "embeddings_created": embeddings_created,
            }

        except FileNotFoundError:
            return {"success": False, "error": f"CSV file not found: {csv_file_path}"}
        except pd.errors.EmptyDataError:
            return {"success": False, "error": "CSV file is empty"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {str(e)}"}

    def get_failure_mode_ids(self) -> list:
        """
        Get all failure mode IDs.

        Returns:
            list: List of failure mode IDs
        """
        try:
            result = self.query("MATCH (fd:FailureMode) RETURN ID(fd)")
            return [record["ID(fd)"] for record in result]
        except Exception as e:
            print(f"Error getting failure mode IDs: {e}")
            return []

    def traverse_graph(self, failure_mode_id: int) -> list:
        """
        Traverse the graph for a given failure mode ID.

        Args:
            failure_mode_id: The failure mode ID to traverse

        Returns:
            list: List of connected nodes
        """
        try:
            query = self.TRAVERSE_QUERY.format(id=failure_mode_id)
            return self.query(query)
        except Exception as e:
            print(f"Error traversing graph: {e}")
            return []

    def create_chunk(self, nodes: list) -> tuple[str, dict]:
        """
        Create a text chunk from a list of nodes for embedding.

        Args:
            nodes: List of node dictionaries

        Returns:
            tuple: (chunk_text, node_ids_dict)
        """
        fm, fc, fe, fd, ps = [[] for _ in range(5)]

        node_ids = {
            "failureModeIds": [],
            "failureEffectIds": [],
            "failureCauseIds": [],
            "failureMeasureIds": [],
            "processStepIds": [],
        }

        for node in nodes:
            if node["fm"] not in fm:
                fm.append(node["fm"])
                node_ids["failureMeasureIds"].append(node["ID(fm)"])
            if node["fc"] not in fc:
                fc.append(node["fc"])
                node_ids["failureCauseIds"].append(node["ID(fc)"])
            if node["fe"] not in fe:
                fe.append(node["fe"])
                node_ids["failureEffectIds"].append(node["ID(fe)"])
            if node["fd"] not in fd:
                fd.append(node["fd"])
                node_ids["failureModeIds"].append(node["ID(fd)"])
            if node["ps"] not in ps:
                ps.append(node["ps"])
                node_ids["processStepIds"].append(node["ID(ps)"])

        chunk = (
            ", ".join("ProcessStep: " + i["ProcessStep"] for i in ps)
            + "".join(
                ", FailureMode: " + i["FailureMode"] + ", RPN: " + str(i["RPN"])
                for i in fd
            )
            + "".join(
                ", FailureEffect: " + i["FailureEffect"] + ", S: " + str(i["S"])
                for i in fe
            )
            + "".join(
                ", FailureCause: " + i["FailureCause"] + ", O: " + str(i["O"])
                for i in fc
            )
            + "".join(
                ", FailureMeasure: "
                + i["FailureMeasure"]
                + ", DetectionMeasure: "
                + i["DetectionMeasure"]
                + ", D: "
                + str(i["D"])
                for i in fm
            )
        )

        return chunk, node_ids

    def create_new_index(self) -> bool:
        """
        Create a new vector index for FMEA embeddings.

        Returns:
            bool: True if successful
        """
        try:
            # Create vector index
            index_query = """
            CREATE VECTOR INDEX fmea_embeddings IF NOT EXISTS
            FOR (c:Chunk)
            ON c.embedding
            OPTIONS {indexConfig: {
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'
            }}
            """
            self.query(index_query)
            return True
        except Exception as e:
            print(f"Error creating index: {e}")
            return False

    def retrieve_existing_index(self) -> Optional[int]:
        """
        Check if the vector index already exists.

        Returns:
            Optional[int]: Embedding dimension if exists, None otherwise
        """
        try:
            result = self.query("SHOW INDEXES")
            for record in result:
                if record.get("name") == "fmea_embeddings":
                    return 1536  # OpenAI embedding dimension
            return None
        except Exception as e:
            print(f"Error checking index: {e}")
            return None

    def create_vector_embeddings(self) -> dict:
        """
        Create vector embeddings for all failure modes in the graph.

        Returns:
            dict: Status and count of embeddings created
        """
        if not self.enable_embeddings or not self.vector_store:
            return {
                "success": False,
                "error": "Embeddings are not enabled. Initialize with enable_embeddings=True",
            }

        try:
            # Get all failure mode IDs
            failure_mode_ids = self.get_failure_mode_ids()

            if not failure_mode_ids:
                return {"success": True, "embeddings_created": 0}

            # Check if the index already exists
            embedding_dimension = self.retrieve_existing_index()

            # If the index doesn't exist, create it
            if not embedding_dimension:
                self.create_new_index()

            embeddings_created = 0

            # Add embeddings for each failure mode
            for fm_id in failure_mode_ids:
                nodes = self.traverse_graph(fm_id)
                if not nodes:
                    continue

                chunk, node_ids = self.create_chunk(nodes)

                # Add text to vector store
                embedded_node_ids = self.vector_store.add_texts(
                    texts=[chunk], metadatas=[node_ids]
                )

                if embedded_node_ids:
                    embedded_node_id = embedded_node_ids[0]

                    # Create relationship between FailureMode and Chunk
                    query_parts = [
                        self.MATCH_QUERY.format(
                            nodeRef="index",
                            node="Chunk",
                            properties=self.format_properties({"id": embedded_node_id}),
                        ),
                        "WITH index",
                        self.MATCH_QUERY.format(
                            nodeRef="fd", node="FailureMode", properties="{}"
                        ),
                        f"WHERE ID(fd)={fm_id}",
                        self.MERGE_RELATION_QUERY.format(
                            nodeRef1="fd", relation="isIndexed", nodeRef2="index"
                        ),
                    ]

                    try:
                        self.query("\n".join(query_parts))
                        embeddings_created += 1
                    except Exception as e:
                        print(f"Error linking embedding for failure mode {fm_id}: {e}")

            return {"success": True, "embeddings_created": embeddings_created}

        except Exception as e:
            return {"success": False, "error": f"Error creating embeddings: {str(e)}"}

    def similarity_search(self, query: str, k: int = 3) -> list:
        """
        Perform similarity search on vector embeddings.

        Args:
            query: The search query
            k: Number of results to return

        Returns:
            list: List of similar chunks
        """
        if not self.enable_embeddings or not self.vector_store:
            return []

        try:
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"Error performing similarity search: {e}")
            return []

    def get_graph_stats(self) -> dict:
        """
        Get statistics about the FMEA graph.

        Returns:
            dict: Graph statistics
        """
        try:
            stats = {}

            # Count nodes by type
            node_types = [
                "FailureMode",
                "ProcessStep",
                "FailureEffect",
                "FailureCause",
                "FailureMeasure",
            ]

            for node_type in node_types:
                result = self.query(f"MATCH (n:{node_type}) RETURN count(n) as count")
                stats[f"{node_type}_count"] = result[0]["count"] if result else 0

            # Count relationships
            rel_result = self.query("MATCH ()-[r]->() RETURN count(r) as count")
            stats["relationship_count"] = rel_result[0]["count"] if rel_result else 0

            return {"success": True, "stats": stats}

        except Exception as e:
            return {"success": False, "error": str(e)}
