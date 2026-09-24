import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# Set up and load your env parameters and instantiate your model.
dotenv.load_dotenv()

from smolagents import tool, CodeAgent, ToolCallingAgent, OpenAIServerModel

# Use Vocareum OpenAI endpoint
model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_base="https://openai.vocareum.com/v1",
    api_key=os.environ.get("OPENAI_API_KEY", os.environ.get("API_KEY", "")),
)

"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""


# ============================================================================
# STUDENT INSTRUCTIONS - READ ME FIRST
# ============================================================================
# Everything ABOVE this point is provided for you and should NOT be changed:
#   - the paper_supplies catalog
#   - all of the database helper functions (get_stock_level, get_cash_balance,
#     create_transaction, get_supplier_delivery_date, generate_financial_report,
#     search_quote_history, etc.)
#   - the model setup and the _resolve_item_name helper just below
#
# Your job is to complete the parts marked  # TODO  below. They are:
#   1. Five of the tool functions (check_item_stock, reorder_stock,
#      get_quote_history, generate_quote, fulfill_order). One worked example
#      (check_inventory) and two simple ones (check_cash_balance,
#      get_financial_summary) are already done so you have a pattern to follow.
#   2. The quoting_agent and sales_agent (inventory_agent is done as an example).
#   3. The orchestrator_agent that manages the three specialist agents.
#
# The test harness at the bottom (run_test_scenarios) is complete. Once your
# orchestrator_agent is built, running this file will exercise the whole system.
#
# TIP: each tool just wraps the database helpers above and returns a readable
# string. Look at how check_inventory does it before writing your own.
# ============================================================================


# --- Helper: fuzzy item name resolver (PROVIDED - do not modify) ---
def _resolve_item_name(query: str) -> str:
    """
    Resolve a customer's description of an item to the exact catalog item_name.
    Uses substring and keyword matching against both paper_supplies and inventory.
    Returns the best-matching canonical item name, or None if no match found.
    """
    query_lower = query.lower().strip()

    # Build list of all known item names (from paper_supplies which is the full catalog)
    all_items = [p["item_name"] for p in paper_supplies]

    # 1. Exact match (case-insensitive)
    for name in all_items:
        if name.lower() == query_lower:
            return name

    # 2. Check if query is contained in a catalog name or vice versa
    candidates = []
    for name in all_items:
        name_lower = name.lower()
        if query_lower in name_lower or name_lower in query_lower:
            candidates.append(name)

    if len(candidates) == 1:
        return candidates[0]

    # 3. Keyword overlap scoring
    query_words = set(query_lower.replace("-", " ").replace(",", " ").split())
    # Remove filler words
    filler = {"sheets", "of", "high", "quality", "in", "various", "colors", "assorted",
              "size", "white", "the", "a", "an", "for", "and", "per", "rolls", "roll",
              "x", "inch", "inches", "reams", "ream", "colorful", "sturdy"}
    query_keywords = query_words - filler

    best_score = 0
    best_match = None
    for name in all_items:
        name_words = set(name.lower().replace("-", " ").replace("(", " ").replace(")", " ").split())
        overlap = len(query_keywords & name_words)
        if overlap > best_score:
            best_score = overlap
            best_match = name

    if best_score > 0:
        return best_match

    return None


# Tools for inventory agent

@tool
def check_inventory(as_of_date: str) -> str:
    """
    Check the full inventory status as of a given date. Returns all items with their current stock levels
    and flags items that are below their minimum stock level and need reordering.

    Args:
        as_of_date: ISO-formatted date string (YYYY-MM-DD) to check inventory as of.

    Returns:
        A formatted string listing all inventory items, their stock, min levels, and reorder flags.
    """
    # NOTE (PROVIDED EXAMPLE): study this tool. Your TODO tools follow the same shape:
    # call one or more of the database helpers above, then return a readable string.
    inventory = get_all_inventory(as_of_date)
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)

    result_lines = ["Current Inventory Status:"]
    reorder_needed = []

    for _, item in inventory_df.iterrows():
        name = item["item_name"]
        stock = inventory.get(name, 0)
        min_level = item["min_stock_level"]
        unit_price = item["unit_price"]
        status = "OK"
        if stock <= min_level:
            status = "LOW - REORDER NEEDED"
            reorder_needed.append(name)
        result_lines.append(
            f"  - {name}: stock={int(stock)}, min_level={int(min_level)}, "
            f"unit_price=${unit_price:.2f}, status={status}"
        )

    if reorder_needed:
        result_lines.append(f"\nItems needing reorder: {', '.join(reorder_needed)}")
    else:
        result_lines.append("\nAll items are adequately stocked.")

    return "\n".join(result_lines)


@tool
def check_item_stock(item_name: str, as_of_date: str) -> str:
    """
    Check the stock level and details for a specific item as of a given date.
    The item_name will be fuzzy-matched to the closest item in the catalog.
    Use EXACT catalog names when possible. Available catalog items include:
    A4 paper, Letter-sized paper, Cardstock, Colored paper, Glossy paper, Matte paper,
    Recycled paper, Eco-friendly paper, Poster paper, Banner paper, Kraft paper,
    Construction paper, Wrapping paper, Glitter paper, Decorative paper, Letterhead paper,
    Legal-size paper, Crepe paper, Photo paper, Uncoated paper, Butcher paper,
    Heavyweight paper, Standard copy paper, Bright-colored paper, Patterned paper,
    Paper plates, Paper cups, Paper napkins, Disposable cups, Table covers, Envelopes,
    Sticky notes, Notepads, Invitation cards, Flyers, Party streamers,
    Decorative adhesive tape (washi tape), Paper party bags, Name tags with lanyards,
    Presentation folders, Large poster paper (24x36 inches),
    Rolls of banner paper (36-inch width), 100 lb cover stock, 80 lb text paper,
    250 gsm cardstock, 220 gsm poster paper.

    Args:
        item_name: The name of the inventory item to check (will be fuzzy-matched).
        as_of_date: ISO-formatted date string (YYYY-MM-DD).

    Returns:
        A string with item stock level, unit price, min stock level, and whether reorder is needed.
    """
    # TODO: Implement this tool. Report the stock level and details for one item.
    # Hint: First map the input to a real catalog name with _resolve_item_name(item_name).
    # Hint: Get current stock with get_stock_level(item_name, as_of_date). It returns a
    #       DataFrame; read the value with df["current_stock"].iloc[0] (guard for empty).
    # Hint: Look up unit_price and min_stock_level from the `inventory` table, e.g.
    #       pd.read_sql("SELECT * FROM inventory WHERE item_name = :name", db_engine,
    #                   params={"name": item_name})
    # Hint: An item can exist in paper_supplies but not be carried in inventory (stock 0).
    #       Handle that case, and the "not found anywhere" case, with clear messages.
    # Hint: Flag reorder when stock <= min_stock_level. Return a readable multi-line string.
    resolved = _resolve_item_name(item_name)
    if resolved is None:
        return f"'{item_name}' is not in the catalog and is unavailable."

    stock_df = get_stock_level(resolved, as_of_date)
    stock = int(stock_df["current_stock"].iloc[0]) if not stock_df.empty else 0
    inv_df = pd.read_sql("SELECT * FROM inventory WHERE item_name = :name",
                         db_engine, params={"name": resolved})

    if inv_df.empty:
        unit_price = next(p["unit_price"] for p in paper_supplies if p["item_name"] == resolved)
        return (f"{resolved}: not carried in inventory (stock 0) as of {as_of_date}.\n"
                f"  Catalog price ${unit_price:.2f}/unit - must be ordered from the supplier.")

    unit_price = float(inv_df["unit_price"].iloc[0])
    min_level = int(inv_df["min_stock_level"].iloc[0])
    return (f"{resolved} as of {as_of_date}:\n"
            f"  stock={stock}, min_level={min_level}, unit_price=${unit_price:.2f}\n"
            f"  reorder needed: {'YES' if stock <= min_level else 'NO'}")
    # raise NotImplementedError("TODO: implement check_item_stock")


@tool
def reorder_stock(item_name: str, quantity: int, order_date: str) -> str:
    """
    Place a reorder for an item from the supplier. This creates a stock_orders transaction
    and deducts the cost from cash balance. Checks cash availability before ordering.
    The item_name will be fuzzy-matched to the closest item in the catalog.

    Args:
        item_name: The name of the item to reorder (fuzzy-matched to catalog).
        quantity: Number of units to order.
        order_date: ISO-formatted date string (YYYY-MM-DD) for the order.

    Returns:
        A string confirming the reorder or explaining why it cannot be fulfilled.
    """
    # TODO: Implement this tool. Place a supplier reorder and record it as a transaction.
    # Hint: Resolve the name first with _resolve_item_name(item_name).
    # Hint: Find the unit_price (from the `inventory` table, or fall back to paper_supplies).
    #       If it is in neither, return an error string.
    # Hint: total_cost = quantity * unit_price.
    # Hint: Check funds with get_cash_balance(order_date). If total_cost > cash, refuse.
    # Hint: Get the arrival date with get_supplier_delivery_date(order_date, quantity).
    # Hint: Record it: create_transaction(item_name, "stock_orders", quantity, total_cost,
    #       delivery_date). Return a confirmation string including the transaction id.
    # raise NotImplementedError("TODO: implement reorder_stock")

    resolved = _resolve_item_name(item_name)
    if resolved is None:
        return f"Cannot reorder: '{item_name}' is not in the catalog."

    inv_df = pd.read_sql("SELECT * FROM inventory WHERE item_name = :name",
                         db_engine, params={"name": resolved})
    unit_price = float(inv_df["unit_price"].iloc[0]) if not inv_df.empty else next(
        (p["unit_price"] for p in paper_supplies if p["item_name"] == resolved), None)
    if unit_price is None:
        return f"Cannot reorder: no price on record for '{resolved}'."

    total_cost = quantity * unit_price
    cash = get_cash_balance(order_date)
    if total_cost > cash:
        return (f"Cannot reorder {quantity} units of {resolved}: cost ${total_cost:.2f} exceeds "
                f"the cash balance of ${cash:.2f} as of {order_date}.")

    delivery_date = get_supplier_delivery_date(order_date, quantity)
    tx_id = create_transaction(resolved, "stock_orders", quantity, total_cost, delivery_date)
    return (f"Reordered {quantity} units of {resolved} for ${total_cost:.2f} (transaction {tx_id}). "
            f"Expected delivery {delivery_date}. Cash remaining: ${cash - total_cost:.2f}")

# Tools for quoting agent

@tool
def get_quote_history(search_terms: str) -> str:
    """
    Search historical quotes by keywords to find relevant past pricing. Helps determine
    competitive and consistent pricing for new quote requests.

    Args:
        search_terms: Comma-separated search keywords (e.g. "wedding,invitation,cards").

    Returns:
        A formatted string of matching historical quotes with amounts and explanations.
    """
    # TODO: Implement this tool. Search past quotes and return them as a readable string.
    # Hint: search_terms arrives as a comma-separated string. Split it into a list of terms
    #       (strip whitespace, drop empties).
    # Hint: Call search_quote_history(terms, limit=5). It returns a list of dicts with keys
    #       like 'original_request', 'total_amount', 'quote_explanation', 'job_type',
    #       'order_size', 'event_type'.
    # Hint: If the list is empty, say so. Otherwise format each result into a few lines.
    # raise NotImplementedError("TODO: implement get_quote_history")

    terms = [t.strip() for t in search_terms.split(",") if t.strip()]
    results = search_quote_history(terms, limit=5)
    if not results:
        return f"No historical quotes found for: {', '.join(terms)}"

    result_lines = [f"{len(results)} historical quote(s) for: {', '.join(terms)}"]
    for q in results:
        result_lines.append(
            f"\n  Amount: ${float(q['total_amount']):.2f} | job={q['job_type']} | "
            f"size={q['order_size']} | event={q['event_type']}\n"
            f"  Request: {str(q['original_request'])[:150]}\n"
            f"  Reasoning: {str(q['quote_explanation'])[:200]}")
    return "\n".join(result_lines)

@tool
def generate_quote(item_name: str, quantity: int, as_of_date: str) -> str:
    """
    Generate a price quote for a customer based on item, quantity, and current inventory.
    Applies bulk discounts: 5% for 100-499 units, 10% for 500-999, 15% for 1000+.
    The item_name will be fuzzy-matched to the closest item in the catalog.

    Args:
        item_name: The name of the item the customer wants (fuzzy-matched).
        quantity: The number of units requested.
        as_of_date: ISO-formatted date (YYYY-MM-DD) for the quote.

    Returns:
        A formatted quote string including pricing, discounts, availability, and delivery estimate.
    """
    # TODO: Implement this tool. Produce a customer price quote for one item.
    # Hint: Resolve the name with _resolve_item_name(item_name).
    # Hint: Find unit_price and current stock. If the item is known but not stocked,
    #       treat current stock as 0 (it can still be quoted and ordered in).
    # Hint: Apply bulk discount tiers based on quantity:
    #         >= 1000  -> 15%
    #         >= 500   -> 10%
    #         >= 100   -> 5%
    #         otherwise 0%
    # Hint: base_price = quantity * unit_price; subtract the discount to get the final price.
    # Hint: Decide availability: fully in stock (deliver on as_of_date), partial (compute the
    #       shortfall and use get_supplier_delivery_date(as_of_date, shortfall)), or out of
    #       stock (order the full quantity). Return a clear, formatted quote string.
    # raise NotImplementedError("TODO: implement generate_quote")
    resolved = _resolve_item_name(item_name)
    if resolved is None:
        return f"Cannot quote '{item_name}': not available in our catalog."

    inv_df = pd.read_sql("SELECT * FROM inventory WHERE item_name = :name",
                         db_engine, params={"name": resolved})
    unit_price = float(inv_df["unit_price"].iloc[0]) if not inv_df.empty else next(
        (p["unit_price"] for p in paper_supplies if p["item_name"] == resolved), None)
    if unit_price is None:
        return f"Cannot quote '{item_name}': no price on record for '{resolved}'."

    stock_df = get_stock_level(resolved, as_of_date)
    stock = int(stock_df["current_stock"].iloc[0]) if not stock_df.empty else 0

    discount = 0.15 if quantity >= 1000 else 0.10 if quantity >= 500 else 0.05 if quantity >= 100 else 0.0
    base_price = quantity * unit_price
    total = base_price * (1 - discount)

    if stock >= quantity:
        availability, delivery_date = f"in stock ({stock} units on hand)", as_of_date
    else:
        shortfall = quantity - stock
        delivery_date = get_supplier_delivery_date(as_of_date, shortfall)
        availability = f"{stock} in stock, {shortfall} to be ordered from the supplier"

    return (f"QUOTE for {quantity} units of {resolved} (as of {as_of_date}):\n"
            f"  Unit price ${unit_price:.2f} -> subtotal ${base_price:.2f}\n"
            f"  Bulk discount {discount * 100:.0f}% -> TOTAL ${total:.2f}\n"
            f"  Availability: {availability}\n"
            f"  Estimated delivery: {delivery_date}")

# Tools for ordering agent

@tool
def fulfill_order(item_name: str, quantity: int, order_date: str) -> str:
    """
    Fulfill a customer order by recording a sales transaction. Checks stock availability
    and cash implications. If stock is insufficient, reorders from supplier first.
    The item_name will be fuzzy-matched to the closest item in the catalog.

    Args:
        item_name: The name of the item being sold (fuzzy-matched).
        quantity: The number of units the customer wants to buy.
        order_date: ISO-formatted date (YYYY-MM-DD) for the sale.

    Returns:
        A string confirming the order fulfillment or explaining why it cannot be completed.
    """
    # TODO: Implement this tool. Record a sale, reordering stock first if needed.
    # This is the most involved tool. A suggested sequence:
    #   1. Resolve the name with _resolve_item_name(item_name).
    #   2. Find unit_price and current stock (known-but-not-stocked items start at 0).
    #   3. Compute sale_price using the SAME bulk-discount tiers as generate_quote.
    #   4. If current stock < quantity:
    #        - shortfall = quantity - current_stock; reorder_cost = shortfall * unit_price
    #        - if get_cash_balance(order_date) < reorder_cost, refuse with a clear message
    #        - otherwise record the restock:
    #            delivery_date = get_supplier_delivery_date(order_date, shortfall)
    #            create_transaction(item_name, "stock_orders", shortfall, reorder_cost, delivery_date)
    #          and set fulfillment_date = delivery_date
    #      else: fulfillment_date = order_date
    #   5. Record the sale:
    #        create_transaction(item_name, "sales", quantity, sale_price, fulfillment_date)
    #   6. Return a confirmation string (item, quantity, total sale, fulfillment date, tx id).
    # raise NotImplementedError("TODO: implement fulfill_order")

    resolved = _resolve_item_name(item_name)
    if resolved is None:
        return f"Cannot fulfill: '{item_name}' is not available in our catalog."

    inv_df = pd.read_sql("SELECT * FROM inventory WHERE item_name = :name",
                         db_engine, params={"name": resolved})
    unit_price = float(inv_df["unit_price"].iloc[0]) if not inv_df.empty else next(
        (p["unit_price"] for p in paper_supplies if p["item_name"] == resolved), None)
    if unit_price is None:
        return f"Cannot fulfill: no price on record for '{resolved}'."

    stock_df = get_stock_level(resolved, order_date)
    stock = int(stock_df["current_stock"].iloc[0]) if not stock_df.empty else 0

    discount = 0.15 if quantity >= 1000 else 0.10 if quantity >= 500 else 0.05 if quantity >= 100 else 0.0
    sale_price = quantity * unit_price * (1 - discount)

    restock = ""
    if stock < quantity:
        shortfall = quantity - stock
        reorder_cost = shortfall * unit_price
        cash = get_cash_balance(order_date)
        if cash < reorder_cost:
            return (f"Cannot fulfill {quantity} units of {resolved}: only {stock} in stock and "
                    f"restocking {shortfall} units costs ${reorder_cost:.2f}, over the "
                    f"${cash:.2f} cash balance. Not available in this quantity.")
        fulfillment_date = get_supplier_delivery_date(order_date, shortfall)
        create_transaction(resolved, "stock_orders", shortfall, reorder_cost, fulfillment_date)
        restock = f" (restocked {shortfall} units for ${reorder_cost:.2f})"
    else:
        fulfillment_date = order_date

    tx_id = create_transaction(resolved, "sales", quantity, sale_price, fulfillment_date)
    return (f"ORDER FULFILLED: {quantity} units of {resolved} for ${sale_price:.2f} "
            f"({discount * 100:.0f}% bulk discount), fulfillment date {fulfillment_date}"
            f"{restock}. Transaction {tx_id}.")

@tool
def check_cash_balance(as_of_date: str) -> str:
    """
    Check the current cash balance of the company as of a given date.

    Args:
        as_of_date: ISO-formatted date string (YYYY-MM-DD).

    Returns:
        A string showing the current cash balance.
    """
    # (PROVIDED EXAMPLE - a minimal tool: call a helper, return a string.)
    balance = get_cash_balance(as_of_date)
    return f"Cash balance as of {as_of_date}: ${balance:.2f}"


@tool
def get_financial_summary(as_of_date: str) -> str:
    """
    Generate a financial summary report including cash balance, inventory value,
    total assets, and top-selling products.

    Args:
        as_of_date: ISO-formatted date string (YYYY-MM-DD).

    Returns:
        A formatted financial summary string.
    """
    # (PROVIDED EXAMPLE - wraps generate_financial_report and formats it.)
    report = generate_financial_report(as_of_date)
    lines = [
        f"=== Financial Report as of {as_of_date} ===",
        f"  Cash Balance: ${report['cash_balance']:.2f}",
        f"  Inventory Value: ${report['inventory_value']:.2f}",
        f"  Total Assets: ${report['total_assets']:.2f}",
        f"\n  Top Selling Products:"
    ]
    for p in report["top_selling_products"]:
        if p.get("item_name"):
            lines.append(f"    - {p['item_name']}: {p.get('total_units', 0)} units, ${p.get('total_revenue', 0):.2f} revenue")
    return "\n".join(lines)


# --- Build the catalog string for agent system prompts (PROVIDED - do not modify) ---
_catalog_names = [p["item_name"] for p in paper_supplies]
_catalog_list_str = ", ".join(_catalog_names)

_agent_catalog_note = (
    "IMPORTANT RULES:\n"
    "1. The request includes a 'Date of request' in YYYY-MM-DD format. You MUST use that date "
    "for ALL tool calls (as_of_date / order_date). NEVER use any other date.\n"
    "2. Customer descriptions must be mapped to EXACT catalog item names. The full catalog is:\n"
    f"   {_catalog_list_str}\n"
    "3. Common mappings: 'colored paper' -> 'Colored paper', 'cardstock' -> 'Cardstock', "
    "'washi tape' -> 'Decorative adhesive tape (washi tape)', 'construction paper' -> 'Construction paper', "
    "'glossy paper' -> 'Glossy paper', 'poster paper' -> 'Poster paper', 'recycled paper' -> 'Recycled paper', "
    "'copy paper' or 'printer paper' -> 'Standard copy paper', 'A4 paper' -> 'A4 paper', "
    "'matte paper' -> 'Matte paper', 'photo paper' -> 'Photo paper', 'napkins' -> 'Paper napkins', "
    "'cups' -> 'Paper cups', 'plates' -> 'Paper plates', 'poster board' or 'large poster' -> "
    "'Large poster paper (24x36 inches)', 'banner paper rolls' -> 'Rolls of banner paper (36-inch width)', "
    "'streamers' -> 'Party streamers', 'envelopes' -> 'Envelopes', 'flyers' -> 'Flyers', "
    "'heavyweight' or 'heavy cardstock' -> 'Heavyweight paper', "
    "'invitation cards' -> 'Invitation cards', 'notepads' -> 'Notepads'.\n"
    "4. Items like 'balloons', 'tickets', 'A3 paper', 'signage cardboard' do NOT exist in the catalog. "
    "If an item truly cannot be mapped, say it's unavailable.\n"
    "5. When processing orders, generate a quote first, then fulfill orders for available items.\n"
    "6. Always try to fulfill as much of the order as possible with available catalog items.\n"
)


# Set up your agents and create an orchestration agent that will manage them.

# Inventory Agent - handles inventory queries and reordering
# (PROVIDED EXAMPLE) Study this agent. Your quoting_agent and sales_agent follow the same
# pattern: pass the relevant tools, the shared `model`, a name, a description, and max_steps.
inventory_agent = ToolCallingAgent(
    tools=[check_inventory, check_item_stock, reorder_stock],
    model=model,
    name="inventory_agent",
    description=(
        "Manages inventory for the Beaver's Choice Paper Company. "
        "Can check current stock levels for all items or specific items, "
        "identify items that need reordering, and place restock orders with the supplier. "
        "Use this agent for any inventory-related questions or restocking needs. "
        "IMPORTANT: Always pass the request date (YYYY-MM-DD) from the customer request in your task message."
    ),
    max_steps=6,
)

# Quoting Agent - handles quote generation using history and pricing
# TODO: Build the quoting agent (follow the inventory_agent example above).
# Hint: tools should be the quoting-related ones: [get_quote_history, generate_quote, check_item_stock]
# Hint: pass model=model, name="quoting_agent", a short description of what it does, and max_steps=6.
# Hint: in the description, remind it to use the request date and EXACT catalog item names.
quoting_agent = ToolCallingAgent(
    tools=[get_quote_history, generate_quote, check_item_stock],
    model=model,
    name="quoting_agent",
    description=(
        "Generates customer price quotes: checks comparable past quotes and current stock, then "
        "prices the request with bulk discounts and a delivery estimate. "
        "IMPORTANT: always pass the request date (YYYY-MM-DD) and EXACT catalog item names."
    ),
    max_steps=6,
)

# Sales/Order Agent - handles order fulfillment
# TODO: Build the sales agent (same pattern as the others).
# Hint: tools should include fulfillment and checking: [fulfill_order, check_item_stock,
#       check_cash_balance, get_financial_summary]
# Hint: pass model=model, name="sales_agent", a description, and max_steps=8.
sales_agent = ToolCallingAgent(
    tools=[fulfill_order, check_item_stock, check_cash_balance, get_financial_summary],
    model=model,
    name="sales_agent",
    description=(
        "Fulfills customer orders by recording sales, restocking from the supplier when stock is "
        "short, and reporting cash or financial summaries. "
        "IMPORTANT: always pass the request date (YYYY-MM-DD) as order_date and EXACT catalog "
        "item names; fulfill one item per call and report anything unfulfilled and why."
    ),
    max_steps=8,
)
# Orchestrator Agent - delegates to the specialist agents
# TODO: Build the orchestrator. This is the agent the test harness calls.
# Hints:
#   - It has NO tools of its own: tools=[]
#   - It manages the specialists: managed_agents=[inventory_agent, quoting_agent, sales_agent]
#   - pass model=model and name="orchestrator_agent"
#   - Its description should INCLUDE the provided _agent_catalog_note (so it knows the date rule
#     and item-name mappings) PLUS a short numbered workflow, for example:
#       1) ask quoting_agent to quote each requested item (pass the exact request date),
#       2) ask sales_agent to fulfill the items that can be fulfilled,
#       3) ask inventory_agent to reorder anything that needs restocking first,
#       4) compile a customer-facing reply with prices, status, delivery dates, and any
#          items that could not be fulfilled and why.
#   - Use max_steps=10.
#   - Keep the variable name `orchestrator_agent` (run_test_scenarios below calls it).
orchestrator_agent = ToolCallingAgent(
    tools=[],
    model=model,
    managed_agents=[inventory_agent, quoting_agent, sales_agent],
    name="orchestrator_agent",
    description=_agent_catalog_note + (
        "\nWORKFLOW:\n"
        "1. Ask quoting_agent to quote each requested item, passing the exact request date.\n"
        "2. Ask sales_agent to fulfill the items that can be fulfilled.\n"
        "3. Ask inventory_agent to reorder anything that needs restocking first, then retry.\n"
        "4. Reply with per-item price, status, delivery date, and any item not fulfilled and why."
    ),
    max_steps=10,
    max_tool_threads=1 #fixes issue with max steps + 1 being called twice in parallel workflows. 
)

# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############

    # The orchestrator_agent is already defined at module level.
    # We use it directly to process each request.

    results = []
    prev_cash = current_cash
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        try:
            response = orchestrator_agent.run(request_with_date)
            status = "error" if "cannot" in str(response).lower() and "not available" in str(response).lower() else "fulfilled"
        except Exception as e:
            print(f"Error processing request: {e}")
            response = f"Error: {str(e)}"
            status = "error"

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]
        cash_change = current_cash - prev_cash

        # Determine fulfillment status based on cash change
        if status != "error":
            if cash_change > 0:
                status = "fulfilled"
            elif cash_change == 0 and "not available" in str(response).lower():
                status = "unfulfilled"
            else:
                status = "fulfilled"

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        # Truncate response for CSV readability (keep first 300 chars)
        response_str = str(response).replace("\n", " ").replace("\r", " ")
        response_summary = response_str[:300] + ("..." if len(response_str) > 300 else "")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "job": row["job"],
                "event": row["event"],
                "status": status,
                "cash_balance": round(current_cash, 2),
                "cash_change": round(cash_change, 2),
                "inventory_value": round(current_inventory, 2),
                "response_summary": response_summary,
            }
        )

        prev_cash = current_cash
        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
