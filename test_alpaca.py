from dotenv import load_dotenv
import os
from alpaca.trading.client import TradingClient

load_dotenv(override=True)

client = TradingClient(
    os.environ["ALPACA_API_KEY"],
    os.environ["ALPACA_SECRET_KEY"],
    paper=True
)

account = client.get_account()
print(f"Account status: {account.status}")
print(f"Buying power:  ${float(account.buying_power):,.2f}")
print(f"Equity:        ${float(account.equity):,.2f}")
