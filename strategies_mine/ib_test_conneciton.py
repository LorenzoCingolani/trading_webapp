from ib_insync import IB

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # or 4002 if using IB Gateway

print("Connected:", ib.isConnected())

ib.disconnect()
