import streamlit as st
import pandas as pd
import plotly.express as px
import config as cf

st.set_page_config(page_title="E-Commerce Dashboard", layout="wide")

# Load datasets
@st.cache_data
def load_data():
    customers = pd.read_csv(cf.CUSTOMERS_PATH)
    geolocation = pd.read_csv(cf.GEOLOCATION_PATH)
    order_items = pd.read_csv(cf.ORDER_ITEMS_PATH)
    order_payments = pd.read_csv(cf.ORDER_PAYMENTS_PATH)
    order_reviews = pd.read_csv(cf.ORDER_REVIEWS_DATASET)
    orders = pd.read_csv(cf.ORDERS_PATH)
    product_categories = pd.read_csv(cf.PRODUCT_CATEGORY_PATH)
    products = pd.read_csv(cf.PRODUCTS_PATH)
    sellers = pd.read_csv(cf.SELLERS_PATH)
    return customers, geolocation, order_items, order_payments, order_reviews, orders, product_categories, products, sellers

# Load data
customers, geolocation, order_items, order_payments, order_reviews, orders, product_categories, products, sellers = load_data()

# Streamlit layout
st.title("📊 E-Commerce Dashboard")

overview_tb, products_tb, sellers_tb, delivery_tb, transact_tb = st.tabs(["📈 Overview", "📦 Products", "📌 Sellers", "🚚 Deliveries", "💳 Transaction"])

# # Overview metrics
with overview_tb:
    main_col1, main_col2 = st.columns(2, border=True)

    with main_col1:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Orders", f"{orders.shape[0]:,}")
        col2.metric("Total Customers", f"{customers.shape[0]:,}")
        col3.metric("Total Sellers", f"{sellers.shape[0]:,}")
        col4.metric("Total Revenue", f"${order_items.price.sum():,.2f}")

    with main_col2:
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
        last_month = orders['order_purchase_timestamp'].max().replace(day=1) - pd.DateOffset(days=1)
        prev_month = last_month - pd.DateOffset(months=1)

        current_revenue = order_items.merge(orders, on='order_id')
        current_revenue = current_revenue[current_revenue['order_purchase_timestamp'].dt.to_period('M') == last_month.to_period('M')]['price'].sum()
        previous_revenue = order_items.merge(orders, on='order_id')
        previous_revenue = previous_revenue[previous_revenue['order_purchase_timestamp'].dt.to_period('M') == prev_month.to_period('M')]['price'].sum()
        delta_revenue = current_revenue - previous_revenue
        delta_revenue_str = f"${delta_revenue:,.2f} (last month)" if delta_revenue >=0 else f"-${abs((current_revenue - previous_revenue)):,.2f} (last month)"

        current_orders = orders[orders['order_purchase_timestamp'].dt.to_period('M') == last_month.to_period('M')].shape[0]
        previous_orders = orders[orders['order_purchase_timestamp'].dt.to_period('M') == prev_month.to_period('M')].shape[0]

        current_customers = customers[customers['customer_unique_id'].isin(orders[orders['order_purchase_timestamp'].dt.to_period('M') == last_month.to_period('M')]['customer_id'])].shape[0]
        previous_customers = customers[customers['customer_unique_id'].isin(orders[orders['order_purchase_timestamp'].dt.to_period('M') == prev_month.to_period('M')]['customer_id'])].shape[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Current Orders", f"{current_orders:,}", delta=f"{(current_orders - previous_orders)} (last month)")
        col2.metric("Current Customer", f"{current_customers:,}", delta=f"{(current_customers - previous_customers)} (last month)")
        col3.metric("Current Revenue", f"${current_revenue:,.2f}", delta=delta_revenue_str)

    with st.container(border=True):
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
        sales_trend = orders.groupby(orders['order_purchase_timestamp'].dt.date).size().reset_index(name='order_count')
        fig_sales = px.line(sales_trend, x='order_purchase_timestamp', y='order_count', title="Sales Over Time")
        st.plotly_chart(fig_sales, use_container_width=True)

    with st.container(border=True):
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
        revenue_trend = order_items.merge(orders, on='order_id').groupby(orders['order_purchase_timestamp'].dt.date)['price'].sum().reset_index()
        fig_revenue = px.line(revenue_trend, x='order_purchase_timestamp', y='price', title="Revenue Over Time")
        st.plotly_chart(fig_revenue, use_container_width=True)

    col1, col2 = st.columns(2, border=True)
    
    with col1:
        products = products.merge(product_categories, on='product_category_name', how='left')
        category_revenue = order_items.merge(products, on='product_id').groupby('product_category_name_english')['price'].sum().reset_index()
        category_revenue = category_revenue.sort_values(by='price', ascending=False)
        top_10_categories = category_revenue.head(10)
        other_category = pd.DataFrame({'product_category_name_english': ['Other'], 'price': [category_revenue.iloc[10:]['price'].sum()]})
        final_category_revenue = pd.concat([top_10_categories, other_category])
        fig_category_revenue = px.pie(final_category_revenue, names='product_category_name_english', values='price', title="Revenue Contribution by Product Category")
        st.plotly_chart(fig_category_revenue, use_container_width=True)

    with col2:
        top_products = order_items.groupby('product_id')['price'].sum().nlargest(10).reset_index()
        top_products = top_products.merge(products, on='product_id')
        fig_top_products = px.bar(top_products, x='product_id', y='price', title="Top Selling Products")
        st.plotly_chart(fig_top_products, use_container_width=True)

with products_tb:
    selected_category = st.selectbox("Select a Product Category:", options=products['product_category_name_english'].dropna().unique())

    filtered_products = products[products['product_category_name_english'] == selected_category]
    filtered_revenue = order_items[order_items['product_id'].isin(filtered_products['product_id'])]
    filtered_sales = filtered_revenue.merge(orders, on='order_id')

    with st.container(border=True):
        col1, col2 = st.columns(2)
        col1.metric("Total Orders", f"{filtered_sales.shape[0]:,}")
        col2.metric("Total Revenue", f"${filtered_revenue.price.sum():,.2f}")

    with st.container(border=True):
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
        sales_trend = filtered_sales.groupby(orders['order_purchase_timestamp'].dt.date).size().reset_index(name='order_count')
        fig_sales = px.line(sales_trend, x='order_purchase_timestamp', y='order_count', title=f"Sales Over Time in {selected_category}")
        st.plotly_chart(fig_sales, use_container_width=True)

    with st.container(border=True):
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
        revenue_trend = filtered_revenue.merge(orders, on='order_id').groupby(orders['order_purchase_timestamp'].dt.date)['price'].sum().reset_index()
        fig_revenue = px.line(revenue_trend, x='order_purchase_timestamp', y='price', title=f"Revenue Over Time in {selected_category}")
        st.plotly_chart(fig_revenue, use_container_width=True)

    with st.container(border=True):
        # Top selling products in the selected category
        top_products = filtered_revenue.groupby('product_id')['price'].sum().nlargest(10).reset_index()
        top_products = top_products.merge(products, on='product_id')
        fig_top_products = px.bar(top_products, x='product_id', y='price', title=f"Top Selling Products in {selected_category}")

        st.plotly_chart(fig_top_products, use_container_width=True)

    with st.container(border=True):
        filetred_order_reviews = filtered_revenue.merge(order_reviews, on='order_id')
        product_reviews = filetred_order_reviews.groupby('review_score')['review_id'].count().reset_index()
        fig_product_reviews = px.bar(product_reviews, x='review_score', y='review_id', title=f"Overall Product Ratings & Reviews in {selected_category}")
        st.plotly_chart(fig_product_reviews, use_container_width=True)

with sellers_tb:
    df_sellers_geo = sellers.merge(geolocation, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')

    with st.container():
        st.subheader("📍Seller Location")
        df_sellers_geo = df_sellers_geo.dropna(subset=["geolocation_lat", "geolocation_lng"])
        st.map(df_sellers_geo, latitude="geolocation_lat", longitude="geolocation_lng")

    with st.container(border=True):
        sellers_by_state = df_sellers_geo.groupby("geolocation_state").size().reset_index(name='count')
        fig_sellers_state = px.bar(sellers_by_state, x='geolocation_state', y='count', title="Number of Sellers per State")
        st.plotly_chart(fig_sellers_state, use_container_width=True)

with delivery_tb:
    with st.container(border=True):
        st.subheader("📦 Delivery Status")
        delivery_status_counts = orders['order_status'].value_counts().reset_index()
        delivery_status_counts.columns = ['status', 'count']

        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)

        col1.metric("Delivered",delivery_status_counts[delivery_status_counts["status"] == "delivered"]["count"].item())
        col2.metric("Shipped", delivery_status_counts[delivery_status_counts["status"] == "shipped"]["count"].item())
        col3.metric("Canceled",delivery_status_counts[delivery_status_counts["status"] == "canceled"]["count"].item())
        col4.metric("Unavailable",delivery_status_counts[delivery_status_counts["status"] == "unavailable"]["count"].item())
        col5.metric("Invoiced",delivery_status_counts[delivery_status_counts["status"] == "invoiced"]["count"].item())
        col6.metric("Processing", delivery_status_counts[delivery_status_counts["status"] == "processing"]["count"].item())
        col7.metric("Created",delivery_status_counts[delivery_status_counts["status"] == "created"]["count"].item())
        col8.metric("Approved",delivery_status_counts[delivery_status_counts["status"] == "approved"]["count"].item())

    with st.container(border=True):
        orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'], errors='coerce')
        orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'], errors='coerce')

        orders['delivery_delay'] = (orders['order_delivered_customer_date'] - orders['order_estimated_delivery_date']).dt.days
        fig_delay_hist = px.histogram(orders, x='delivery_delay', title="Distribution of Delivery Delays (Days)")
        st.plotly_chart(fig_delay_hist, use_container_width=True)

    with st.container(border=True):
        order_id_search = st.text_input("Search Order by ID", "")
        if order_id_search:
            search_result = orders[orders['order_id'].str.contains(order_id_search, case=False, na=False)]
            st.dataframe(search_result, use_container_width=True)

with transact_tb:
    col1, col2 = st.columns(2, border=True)

    with col1:
        transactions_by_type = order_payments.groupby("payment_type").size().reset_index(name='count')
        fig_transactions = px.bar(transactions_by_type, x='payment_type', y='count', title="Transactions by Payment Type")
        st.plotly_chart(fig_transactions, use_container_width=True)

    with col2:
        transactions_amount = order_payments.groupby("payment_type")['payment_value'].sum().reset_index()
        fig_transactions_amount = px.pie(transactions_amount, names='payment_type', values='payment_value', title="Transaction Value by Payment Type")
        st.plotly_chart(fig_transactions_amount, use_container_width=True)