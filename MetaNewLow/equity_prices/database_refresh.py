import os
import pandas as pd
import datetime
from forexconnect import fxcorepy, ForexConnect
from sqlalchemy import create_engine, MetaData, Table, Column, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import CreateSchema
from sqlalchemy.exc import ProgrammingError

# Configuration for SQLAlchemy
DATABASE_URI = 'postgresql://postgres:pergo1234@34.80.139.235/postgres'
engine = create_engine(DATABASE_URI, echo=False)  # 添加 echo=True 以顯示 SQLAlchemy 的調試信息
Session = sessionmaker(bind=engine)
session = Session()

# Create priceDB schema if it does not exist
def create_schema(engine, schema_name):
    with engine.connect() as connection:
        if not engine.dialect.has_schema(connection, schema_name):
            print(f"Creating schema: {schema_name}")
            connection.execute(CreateSchema(schema_name))

# Function to drop table if it exists
def drop_table(engine, schema_name, table_name):
    metadata = MetaData(schema=schema_name)
    table = Table(table_name, metadata, autoload_with=engine)
    with engine.connect() as connection:
        print(f"Dropping table: {schema_name}.{table_name}")
        table.drop(connection, checkfirst=True)

# Function to fetch prices from FXCM
def fetch_prices_from_fxcm(ticker, interval):
    with ForexConnect() as fx:
        try:
            fx.login(config.YOUR_USERNAME, config.YOUR_PASSWORD, config.YOUR_SERVER_ADDRESS, config.YOUR_TYPE)
            start_date = datetime.datetime.strptime("1990-01-01", '%Y-%m-%d')
            end_date = datetime.datetime.now()
            history = fx.get_history(ticker, interval, start_date, end_date)
            df = pd.DataFrame(history)
            print(df)
            return df
        except Exception as e:
            print(f"Error fetching data from FXCM: {e}")
            return pd.DataFrame()

# Main function to auto-refresh data
def auto_refresh(symbol, interval):
    print(f'目前數據：{symbol}')
    data = fetch_prices_from_fxcm(symbol, interval)
    if data.empty:
        print(f"No data fetched for {symbol} with interval {interval}.")
        return
    
    # Ensure column names are consistent
    data.columns = [col.lower() for col in data.columns]

    # Replace / with _ in the symbol name
    table_name = f"{symbol.replace('/', '_')}_{interval}"

    # Create schema if not exist
    schema_name = 'pricedb'
    create_schema(engine, schema_name)

    # Insert data into the table
    try:
        data.to_sql(table_name, engine, schema=schema_name, if_exists='replace', index=False, method='multi')
        print(f"Data for {symbol} with interval {interval} has been inserted into the table {table_name}.")
    except ProgrammingError as e:
        print(f"Error inserting data into {table_name}")

# Configuration for FXCM
class config(object):
    YOUR_SERVER_ADDRESS = 'fxcorporate.com/Hosts.jsp'
    YOUR_USERNAME = 'D103507351'
    YOUR_PASSWORD = 'yEs0e'
    YOUR_TYPE = 'Demo'

# List of symbols
symbol_list = [ 
                'BTC/USD', 'ETH/USD', 'LTC/USD', 'NAS100', 'VOLX', 'SPX500', 'US30', 'GER30', 'US2000', 'AUS200', 'JPN225', 'USD/JPY', 'XAU/USD', 
                'FRA40', 'UK100', 'CHN50', 'HKG33', 'ESP35', 
                'XAG/USD', 'USOilSpot', 'UKOilSpot', 'AlumSpot', 'LeadSpot', 'NickelSpot', 'ZincSpot',
                'BNB/USD', 'CryptoMajor', 'DOGE/USD', 'ADA/USD', 'AVAX/USD', 'DOT/USD', 'LINK/USD', 'MATIC/USD', 'SOL/USD',
                'AIRLINES', 'ATMX', 'BIOTECH', 'Cryptostock', 'ESPORTS', 'FAANG', 'US.AUTO', 'US.BANKS', 'US.ECOMM',

                
                # 'TSLA', 'NVDA', 'AAPL', 'AMZN', 'AMD', 'NIO', 'GOOG', 'MSFT', 'INTC', 'BABA', 'COIN', 'DIS', 'LCID', 'MARA', 'META', 'PLTR', 'RIVN', 'SQ', 'UBER', 'XPEV',
                ]

symbol_list = [ 'BTC/USD',
                'ETH/USD','NAS100', 'VOLX', 'SPX500', 'US30', 'GER30', 'US2000', 'AUS200', 'JPN225', 'FRA40', 'UK100', 'CHN50', 'HKG33', 'ESP35', 'XAU/USD',
                'XAG/USD', 'USOilSpot', 'UKOilSpot', 'AlumSpot', 'LeadSpot', 'NickelSpot',  'LTC/USD', 
                'BTC/USD',
                'ZincSpot',
                'BNB/USD', 'CryptoMajor', 'DOGE/USD', 'ADA/USD', 'AVAX/USD', 'DOT/USD', 'LINK/USD', 'MATIC/USD', 'SOL/USD',
                'AIRLINES', 'ATMX', 'BIOTECH', 'Cryptostock', 'ESPORTS', 'FAANG', 'US.AUTO', 'US.BANKS', 'US.ECOMM',
                
                'AUD/CAD', 'AUD/CHF', 'AUD/CNH', 'AUD/JPY', 'AUD/NZD', 'AUD/USD', 'CAD/CHF', 'CAD/JPY', 'CHF/JPY', 'EUR/AUD', 'EUR/CAD', 'EUR/CHF', 'EUR/GBP', 'EUR/HUF', 'EUR/JPY', 'EUR/NOK', 'EUR/NZD', 'EUR/SEK', 'EUR/TRY', 'EUR/USD', 
                'GBP/AUD', 'GBP/CAD', 'GBP/CHF', 'GBP/JPY', 'GBP/NZD', 'GBP/USD', 'NZD/CAD', 'NZD/CHF', 'NZD/JPY', 'NZD/USD', 'TRY/JPY', 'USD/CAD', 'USD/CHF', 'USD/CNH', 'USD/DKK', 'USD/HKD', 'USD/HUF', 'USD/JPY', 'USD/MXN', 'USD/NOK', 'USD/PLN', 'USD/SEK', 'USD/TRY', 'USD/ZAR', 'ZAR/JPY',
                
                # 'TSLA', 'NVDA', 'AAPL', 'AMZN', 'AMD', 'NIO', 'GOOG', 'MSFT', 'INTC', 'BABA', 'COIN', 'DIS', 'LCID', 'MARA', 'META', 'PLTR', 'RIVN', 'SQ', 'UBER', 'XPEV',
                ]

# symbol_list = ['BTC/USD', 'NAS100', 'SPX500', 'JPN225']

# Refresh data for each symbol with 1H interval
for symbol in symbol_list:
    auto_refresh(symbol, 'H1')

# Refresh data for each symbol with 1D interval
# for symbol in symbol_list:
#     auto_refresh(symbol, 'D1')

# Function to fetch data from the database
def fetch_from_database(symbol, interval):
    table_name = f"{symbol.replace('/', '_')}_{interval}"
    schema_name = 'pricedb'
    
    with engine.connect() as connection:
        df = pd.read_sql_table(table_name, connection, schema=schema_name)
        df.set_index('date', inplace=True)
        return df

# Example of fetching data from the database
df = fetch_from_database('XAU/USD', 'H1')
print(df.head())
