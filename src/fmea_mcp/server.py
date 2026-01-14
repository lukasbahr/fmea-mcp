"""
FMEA MCP Server

A simple MCP server implementation using FastMCP with Pydantic AI integration.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession
from pydantic import BaseModel, Field

from fmea_mcp.neo4j_service import Neo4JService

# Load environment variables
load_dotenv()

# Create the MCP server
mcp = FastMCP("FMEA MCP Server", json_response=True)


class FMEAAnalysis(BaseModel):
    """FMEA Analysis result structure."""


@mcp.tool()
async def create_fmea_graph_from_csv(
    csv_file_path: str,
    neo4j_url: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    neo4j_database: Optional[str] = None,
    enable_embeddings: bool = False,
    openai_api_key: Optional[str] = None,
    ctx: Context[ServerSession, None] = None,
) -> dict:
    """
    Create an FMEA knowledge graph in Neo4j from a CSV file.

    This tool reads FMEA data from a CSV file and creates a knowledge graph
    with nodes for FailureMode, ProcessStep, FailureEffect, FailureCause,
    and FailureMeasure, along with their relationships.

    Optionally creates vector embeddings for semantic search capabilities.

    Args:
        csv_file_path: Path to the CSV file containing FMEA data
        neo4j_url: Neo4j URL (defaults to NEO4J_URL env var)
        neo4j_user: Neo4j username (defaults to NEO4J_USER env var)
        neo4j_password: Neo4j password (defaults to NEO4J_PASSWORD env var)
        neo4j_database: Neo4j database name (defaults to NEO4J_DATABASE env var or 'neo4j')
        enable_embeddings: Whether to create vector embeddings (requires OpenAI API key)
        openai_api_key: OpenAI API key for embeddings (defaults to OPENAI_API_KEY env var)

    Returns:
        dict: Status and statistics of the graph creation

    Expected CSV format (semicolon-separated):
        ProcessStep;FailureMode;FailureEffect;FailureCause;FailureMeasure;DetectionMeasure;S;O;D;RPN

    Example CSV content:
        ProcessStep;FailureMode;FailureEffect;FailureCause;FailureMeasure;DetectionMeasure;S;O;D;RPN
        Assembly;Bolt missing;Product failure;Operator error;Visual inspection;Check;8;3;2;48
    """
    if ctx:
        await ctx.info(f"Starting FMEA graph creation from: {csv_file_path}")

    # Get Neo4j credentials from environment or parameters
    url = neo4j_url or os.getenv("NEO4J_URL", "bolt://localhost:7687")
    user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
    password = neo4j_password or os.getenv("NEO4J_PASSWORD")
    database = neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")

    if not password:
        error_msg = "Neo4j password not provided. Set NEO4J_PASSWORD environment variable or pass neo4j_password parameter."
        if ctx:
            await ctx.error(error_msg)
        return {"success": False, "error": error_msg}

    # Get OpenAI API key if embeddings are enabled
    api_key = None
    if enable_embeddings:
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            error_msg = "OpenAI API key required for embeddings. Set OPENAI_API_KEY environment variable or pass openai_api_key parameter."
            if ctx:
                await ctx.error(error_msg)
            return {"success": False, "error": error_msg}

    try:
        # Create Neo4j service
        neo4j_service = Neo4JService(
            uri=url,
            username=user,
            password=password,
            database=database,
            enable_embeddings=enable_embeddings,
            openai_api_key=api_key,
        )

        if ctx:
            await ctx.debug(f"Connected to Neo4j at {url}")

        # Create the graph
        result = neo4j_service.create_fmea_graph(csv_file_path)

        # Close connection
        neo4j_service.close()

        if ctx:
            if result.get("success"):
                await ctx.info(
                    f"Successfully created FMEA graph: {result.get('rows_processed', 0)} rows processed"
                )
            else:
                await ctx.error(f"Failed to create FMEA graph: {result.get('error')}")

        return result

    except Exception as e:
        error_msg = f"Error creating FMEA graph: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        return {"success": False, "error": error_msg}


@mcp.tool()
async def get_fmea_graph_stats(
    neo4j_url: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    neo4j_database: Optional[str] = None,
    ctx: Context[ServerSession, None] = None,
) -> dict:
    """
    Get statistics about the FMEA knowledge graph in Neo4j.

    Args:
        neo4j_url: Neo4j URL (defaults to NEO4J_URL env var)
        neo4j_user: Neo4j username (defaults to NEO4J_USER env var)
        neo4j_password: Neo4j password (defaults to NEO4J_PASSWORD env var)
        neo4j_database: Neo4j database name (defaults to NEO4J_DATABASE env var or 'neo4j')

    Returns:
        dict: Graph statistics including node and relationship counts
    """
    if ctx:
        await ctx.info("Fetching FMEA graph statistics")

    # Get Neo4j credentials from environment or parameters
    url = neo4j_url or os.getenv("NEO4J_URL", "bolt://localhost:7687")
    user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
    password = neo4j_password or os.getenv("NEO4J_PASSWORD")
    database = neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")

    if not password:
        error_msg = "Neo4j password not provided. Set NEO4J_PASSWORD environment variable or pass neo4j_password parameter."
        if ctx:
            await ctx.error(error_msg)
        return {"success": False, "error": error_msg}

    try:
        # Create Neo4j service
        neo4j_service = Neo4JService(
            uri=url, username=user, password=password, database=database
        )

        # Get statistics
        result = neo4j_service.get_graph_stats()

        # Close connection
        neo4j_service.close()

        return result

    except Exception as e:
        error_msg = f"Error fetching graph statistics: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        return {"success": False, "error": error_msg}


@mcp.tool()
async def search_fmea_embeddings(
    query: str,
    top_k: int = 3,
    neo4j_url: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    neo4j_database: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    ctx: Context[ServerSession, None] = None,
) -> dict:
    """
    Search FMEA knowledge graph using semantic similarity.

    This tool performs vector similarity search on FMEA embeddings to find
    relevant failure modes, causes, effects, and measures based on a natural
    language query.

    Args:
        query: Natural language search query
        top_k: Number of results to return (default: 3)
        neo4j_url: Neo4j URL (defaults to NEO4J_URL env var)
        neo4j_user: Neo4j username (defaults to NEO4J_USER env var)
        neo4j_password: Neo4j password (defaults to NEO4J_PASSWORD env var)
        neo4j_database: Neo4j database name (defaults to NEO4J_DATABASE env var or 'neo4j')
        openai_api_key: OpenAI API key for embeddings (defaults to OPENAI_API_KEY env var)

    Returns:
        dict: Search results with similar FMEA entries

    Example:
        search_fmea_embeddings("bolt failures in assembly process", top_k=5)
    """
    if ctx:
        await ctx.info(f"Searching FMEA embeddings for: {query}")

    # Get Neo4j credentials from environment or parameters
    url = neo4j_url or os.getenv("NEO4J_URL", "bolt://localhost:7687")
    user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
    password = neo4j_password or os.getenv("NEO4J_PASSWORD")
    database = neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")

    if not password:
        error_msg = "Neo4j password not provided. Set NEO4J_PASSWORD environment variable or pass neo4j_password parameter."
        if ctx:
            await ctx.error(error_msg)
        return {"success": False, "error": error_msg}

    # Get OpenAI API key
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        error_msg = "OpenAI API key required for similarity search. Set OPENAI_API_KEY environment variable or pass openai_api_key parameter."
        if ctx:
            await ctx.error(error_msg)
        return {"success": False, "error": error_msg}

    try:
        # Create Neo4j service with embeddings enabled
        neo4j_service = Neo4JService(
            uri=url,
            username=user,
            password=password,
            database=database,
            enable_embeddings=True,
            openai_api_key=api_key,
        )

        # Perform similarity search
        results = neo4j_service.similarity_search(query, k=top_k)

        # Close connection
        neo4j_service.close()

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append(
                {
                    "content": result.page_content,
                    "metadata": result.metadata if hasattr(result, "metadata") else {},
                }
            )

        return {
            "success": True,
            "query": query,
            "results": formatted_results,
            "count": len(formatted_results),
        }

    except Exception as e:
        error_msg = f"Error searching embeddings: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        return {"success": False, "error": error_msg}


@mcp.tool()
async def answer_fmea_question(
    question: str,
    neo4j_url: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    neo4j_database: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    ctx: Context[ServerSession, None] = None,
) -> dict:
    """
    Answer questions about FMEA data using a hybrid approach.

    This tool uses a sophisticated hybrid RAG approach:
    1. First attempts to generate and execute a Cypher graph query
    2. If unsuccessful, falls back to vector similarity search
    3. Summarizes retrieved context
    4. Generates a comprehensive answer using the context

    This approach combines the precision of graph queries with the flexibility
    of semantic search, ensuring robust question answering even for complex queries.

    Args:
        question: Natural language question about FMEA data
        neo4j_url: Neo4j URL (defaults to NEO4J_URL env var)
        neo4j_user: Neo4j username (defaults to NEO4J_USER env var)
        neo4j_password: Neo4j password (defaults to NEO4J_PASSWORD env var)
        neo4j_database: Neo4j database name (defaults to NEO4J_DATABASE env var or 'neo4j')
        openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)

    Returns:
        dict: Answer with context and metadata including method used (graph_query or vector_search)

    Examples:
        answer_fmea_question("What are the highest risk failure modes in the assembly process?")
        answer_fmea_question("What causes weld failures and how can they be prevented?")
        answer_fmea_question("Which failure modes have RPN above 100?")
    """
    if ctx:
        await ctx.info(f"Answering question: {question}")

    # Get Neo4j credentials from environment or parameters
    url = neo4j_url or os.getenv("NEO4J_URL", "bolt://localhost:7687")
    user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
    password = neo4j_password or os.getenv("NEO4J_PASSWORD")
    database = neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")

    if not password:
        error_msg = "Neo4j password not provided. Set NEO4J_PASSWORD environment variable or pass neo4j_password parameter."
        if ctx:
            await ctx.error(error_msg)
        return {"success": False, "error": error_msg}

    # Get OpenAI API key
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        error_msg = "OpenAI API key required for question answering. Set OPENAI_API_KEY environment variable or pass openai_api_key parameter."
        if ctx:
            await ctx.error(error_msg)
        return {"success": False, "error": error_msg}

    try:
        # Create Neo4j service with embeddings enabled for hybrid approach
        neo4j_service = Neo4JService(
            uri=url,
            username=user,
            password=password,
            database=database,
            enable_embeddings=True,
            openai_api_key=api_key,
        )

        if ctx:
            await ctx.debug(
                "Running hybrid question answering (graph query + vector search fallback)"
            )

        # Run hybrid question answering
        result = neo4j_service.answer_question(question)

        # Close connection
        neo4j_service.close()

        if ctx:
            if result.get("success"):
                method = result.get("method", "unknown")
                await ctx.info(f"Successfully answered question using: {method}")
            else:
                await ctx.error(f"Failed to answer question: {result.get('error')}")

        return result

    except Exception as e:
        error_msg = f"Error answering question: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        return {"success": False, "error": error_msg}


def main():
    """Run the MCP server."""
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
