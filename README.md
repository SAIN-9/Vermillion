# Vermillion

Vermillion is a Streamlit app for exploring retail transaction data. Upload a CSV, XLSX, or Parquet file to view sales, customer, and product summaries, then try the optional customer segmentation, forecast, and basket-analysis tools.

The app expects one row per product on an order.

## Run locally

You need Python 3.10 or later and pip.

```powershell
git clone <your-repository-url>
cd Vermillion
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

Streamlit prints the local address when the app starts, usually `http://localhost:8501`.

## Upload a file

Use a `.csv`, `.xlsx`, or `.parquet` transaction export. Vermillion looks for familiar column names and maps them to its internal names. For the dashboard, include an order ID, customer ID, product description, quantity, unit price, and transaction date. Country is optional; without it, the market chart is skipped.

| Field | Internal name | Example column names |
| --- | --- | --- |
| Order ID | `InvoiceNo` | `invoice`, `order_id`, `transaction_id` |
| Customer ID | `CustomerID` | `customer`, `customer_id`, `client_id` |
| Product code | `StockCode` | `sku`, `item_code`, `product_id` |
| Product description | `Description` | `product`, `product_name`, `item_name` |
| Quantity | `Quantity` | `qty`, `units`, `amount` |
| Unit price | `UnitPrice` | `price`, `unit_price`, `sell_price` |
| Transaction date | `InvoiceDate` | `date`, `order_date`, `purchase_date` |
| Country or region | `Country` | `country`, `region`, `nation` |

After upload, the app checks that the required columns are present and shows a message if it cannot continue. Dates and quantities/prices that cannot be parsed are excluded from analysis. The cleaning option is on by default; it also removes cancelled orders, negative quantities or prices, duplicate rows, and rows with missing order or customer IDs. Turn it off to keep those records where possible.

## Dashboard and analysis

- **Executive Dashboard:** revenue, orders, customers, average order value, repeat-customer rate, revenue over time, leading products and customers, and product revenue concentration.
- **Customer Personas (RFM):** groups customers by recency, order frequency, and revenue using K-Means. Choose three to six groups.
- **Revenue Forecast:** projects daily revenue with Holt-Winters exponential smoothing. Treat it as a rough planning aid; it does not account for promotions, stockouts, or outside events.
- **Market Basket Recommendations:** looks for products that appear together on the same order. Support and confidence are configurable; the analysis considers up to 500 of the most frequently purchased products.
- **Strategic Insights & Export:** shows rule-based observations and lets you download cleaned transactions or, after running RFM, a customer persona CSV.

RFM needs enough distinct customers for the selected number of groups. Forecasts are less useful with short or irregular date histories, and basket analysis can take longer on large files.

## Data handling

Uploaded data is processed in the active Streamlit session; the app does not save it to a database. If you deploy it for other people, use an environment appropriate for the sensitivity of the data.

Uploads are limited to 500 MB by the Streamlit configuration; available memory may impose a lower practical limit. Revenue is displayed with a pound sign (`£`); the app does not convert currencies.

## Project files

```text
app.py                  Streamlit pages and dashboard
ui.py                   Shared presentation helpers
styles.css              Page styling
modules/cleaner.py      Column mapping, data checks, and cleaning
modules/analytics.py    KPIs, charts, and insights
modules/ml_engine.py    RFM, forecasting, and basket analysis
.streamlit/config.toml  Streamlit theme and upload limit
requirements.txt        Python dependencies
```

## Check the Python files

```bash
python -m compileall -q app.py ui.py modules
```
