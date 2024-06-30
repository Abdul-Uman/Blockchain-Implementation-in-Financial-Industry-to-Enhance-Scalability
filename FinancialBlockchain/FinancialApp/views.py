from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import os
from datetime import date
import os
import json
from web3 import Web3, HTTPProvider
import ipfsApi
import os
from django.core.files.storage import FileSystemStorage
import pickle
from datetime import datetime

global details, username
details=''
global contract

api = ipfsApi.Client(host='http://127.0.0.1', port=5001)

def readDetails(contract_type):
    global details
    details = ""
    print(contract_type+"======================")
    blockchain_address = 'http://127.0.0.1:9545' #Blokchain connection IP
    web3 = Web3(HTTPProvider(blockchain_address))
    web3.eth.defaultAccount = web3.eth.accounts[0]
    compiled_contract_path = 'FinancialContract.json' #financial contract code
    deployed_contract_address = '0x8f76904d62A2953472475CB19965E4316bBC5596' #hash address to access student contract
    #deployed_contract_address = '0x1DD4fb45C1cdC8C3f32cbaA60464c8107D4D4058' #hash address to access student contract
    with open(compiled_contract_path) as file:
        contract_json = json.load(file)  # load contract info as JSON
        contract_abi = contract_json['abi']  # fetch contract's abi - necessary to call its functions
    file.close()
    contract = web3.eth.contract(address=deployed_contract_address, abi=contract_abi) #now calling contract to access data
    if contract_type == 'adduser':
        details = contract.functions.getUser().call()
    if contract_type == 'addproduct':
        details = contract.functions.getProducts().call()
    if contract_type == 'addcart':
        details = contract.functions.getCart().call()
    if contract_type == 'addwallet':
        details = contract.functions.getWallets().call()
    if len(details) > 0:
        if 'empty' in details:
            details = details[5:len(details)]       
    print(details)    

def saveDataBlockChain(currentData, contract_type):
    global details
    global contract
    details = ""
    blockchain_address = 'http://127.0.0.1:9545'
    web3 = Web3(HTTPProvider(blockchain_address))
    web3.eth.defaultAccount = web3.eth.accounts[0]
    
    compiled_contract_path = 'FinancialContract.json' #ecommerce contract file
    deployed_contract_address = '0x8f76904d62A2953472475CB19965E4316bBC5596' #contract address
    #deployed_contract_address = '0x1DD4fb45C1cdC8C3f32cbaA60464c8107D4D4058' #contract address
    with open(compiled_contract_path) as file:
        contract_json = json.load(file)  # load contract info as JSON
        contract_abi = contract_json['abi']  # fetch contract's abi - necessary to call its functions
    file.close()
    contract = web3.eth.contract(address=deployed_contract_address, abi=contract_abi)
    readDetails(contract_type)
    if contract_type == 'adduser':
        details+=currentData
        msg = contract.functions.addUser(details).transact()
        tx_receipt = web3.eth.waitForTransactionReceipt(msg)
    if contract_type == 'addproduct':
        details+=currentData
        msg = contract.functions.addProducts(details).transact()
        tx_receipt = web3.eth.waitForTransactionReceipt(msg)
    if contract_type == 'addcart':
        details+=currentData
        msg = contract.functions.addCart(details).transact()
        tx_receipt = web3.eth.waitForTransactionReceipt(msg)
    if contract_type == 'addwallet':
        details+=currentData
        msg = contract.functions.addWallets(details).transact()
        tx_receipt = web3.eth.waitForTransactionReceipt(msg)

def ViewProviders(request):
    if request.method == 'GET':
        global details
        user = ''
        with open("session.txt", "r") as file:
            for line in file:
                user = line.strip('\n')
        file.close()
        output = '<table border=1 align=center>'
        output+='<tr><th><font size=3 color=black>Username</font></th>'
        output+='<th><font size=3 color=black>Contact No</font></th>'
        output+='<th><font size=3 color=black>Email ID</font></th>'
        output+='<th><font size=3 color=black>Address</font></th>'
        output+='<th><font size=3 color=black>User Type</font></th></tr>'
        readDetails("adduser")
        rows = details.split("\n")
        for i in range(len(rows)-1):
            arr = rows[i].split("#")
            if arr[0] == 'signup' and arr[6] == 'Service Provider':
                output+='<tr><td><font size=3 color=black>'+arr[1]+'</font></td>'
                output+='<td><font size=3 color=black>'+arr[3]+'</font></td>'
                output+='<td><font size=3 color=black>'+arr[4]+'</font></td>'
                output+='<td><font size=3 color=black>'+arr[5]+'</font></td>'
                output+='<td><font size=3 color=black>'+arr[6]+'</font></td></tr>'
        output+="</table><br/><br/><br/><br/><br/><br/>"
        context= {'data':output}
        return render(request, 'ViewProviders.html', context)     
        

def getAmount():
    global username
    readDetails("addwallet")
    deposit = 0
    wd = 0
    rows = details.split("\n")
    for i in range(len(rows)-1):
        arr = rows[i].split("#")
        if arr[0] == username:
            if arr[3] == 'Self Deposit':
                deposit = deposit + float(arr[1])
            else:
                wd = wd + float(arr[1])
    deposit = deposit - wd
    return deposit

def AddMoney(request):
    if request.method == 'GET':
        global username
        output = '<tr><td><font size="3" color="black">Username</td><td><input type="text" name="t1" size="20" value="'+username+'" readonly/></td></tr>'
        output += '<tr><td><font size="3" color="black">Available&nbsp;Balance</td><td><input type="text" name="t2" size="20" value='+str(getAmount())+' readonly/></td></tr>'
        context= {'data1':output}
        return render(request, 'AddMoney.html', context) 

def AddMoneyAction(request):
    global details
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        amount = request.POST.get('t3', False)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data = username+"#"+amount+"#"+str(timestamp)+"#Self Deposit\n"
        saveDataBlockChain(data,"addwallet")
        context= {'data':'Money added to user wallet '+username}
        return render(request, 'UserScreen.html', context)

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})    

def BrowseProducts(request):
    if request.method == 'GET':
        output = '<tr><td><font size="" color="black">Product&nbsp;Name</font></td><td><select name="t1">'
        readDetails("addproduct")
        rows = details.split("\n")
        for i in range(len(rows)-1):
            arr = rows[i].split("#")
            if arr[0] == 'addproduct':
                output+='<option value="'+arr[2]+'">'+arr[2]+'</option>'
        output+="</select></td></tr>"
        context= {'data1':output}
        return render(request, 'BrowseProducts.html', context)

def Login(request):
    if request.method == 'GET':
       return render(request, 'Login.html', {})
    
def ViewOrders(request):
    if request.method == 'GET':
        global details
        user = ''
        with open("session.txt", "r") as file:
            for line in file:
                user = line.strip('\n')
        file.close()
        output = '<table border=1 align=center>'
        output+='<tr><th><font size=3 color=black>Product Name</font></th>'
        output+='<th><font size=3 color=black>Customer Name</font></th>'
        output+='<th><font size=3 color=black>Contact No</font></th>'
        output+='<th><font size=3 color=black>Email ID</font></th>'
        output+='<th><font size=3 color=black>Address</font></th>'
        output+='<th><font size=3 color=black>Ordered Date</font></th></tr>'
        readDetails("addcart")
        rows = details.split("\n")
        for i in range(len(rows)-1):
            arr = rows[i].split("#")
            if arr[0] == 'bookorder':
                print(arr[2]+" "+user)
                details = arr[3].split(",")
                pid = arr[1]
                user = arr[2]
                book_date = arr[4]
                output+='<tr><td><font size=3 color=black>'+pid+'</font></td>'
                output+='<td><font size=3 color=black>'+user+'</font></td>'
                output+='<td><font size=3 color=black>'+details[0]+'</font></td>'
                output+='<td><font size=3 color=black>'+details[1]+'</font></td>'
                output+='<td><font size=3 color=black>'+details[2]+'</font></td>'
                output+='<td><font size=3 color=black>'+str(book_date)+'</font></td></tr>'
        output+="</table><br/><br/><br/><br/><br/><br/>"
        context= {'data':output}
        return render(request, 'ViewOrders.html', context)     

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})

def AddProduct(request):
    if request.method == 'GET':
       return render(request, 'AddProduct.html', {})

def BookOrders(request):
    if request.method == 'GET':
        global details
        provider = request.GET['farmer']
        product = request.GET['crop']
        amount = request.GET['amount']
        output = '<tr><td><font size="3" color="black">Service&nbsp;Provider</td><td><input type="text" name="t1" size="20" value="'+provider+'" readonly/></td></tr>'
        output += '<tr><td><font size="3" color="black">Product</td><td><input type="text" name="t2" size="20" value="'+product+'" readonly/></td></tr>'
        output += '<tr><td><font size="3" color="black">Amount</td><td><input type="text" name="t3" size="20" value='+amount+' readonly/></td></tr>'
        context= {'data1':output}
        return render(request, 'BookOrders.html', context)   

def BookOrder(request):
    if request.method == 'POST':
        global details, username
        pid = request.POST.get('t2', False)
        amount = request.POST.get('t3', False)
        payment_option = request.POST.get('t4', False)
        balance = getAmount()
        output = "Insufficient Balance is Wallet"
        details = ""
        if float(amount) < balance and payment_option == 'Wallet':
            readDetails("adduser")
            rows = details.split("\n")
            output = 'Your Order details Updated & payment done from wallet<br/>'
            for i in range(len(rows)-1):
                arr = rows[i].split("#")
                if arr[0] == "signup":
                    if arr[1] == username:
                        details = arr[3]+","+arr[4]+","+arr[5]
                        break
        if payment_option == 'Card':
            readDetails("adduser")
            rows = details.split("\n")
            output = 'Your Order details Updated & payment done from card<br/>'
            for i in range(len(rows)-1):
                arr = rows[i].split("#")
                if arr[0] == "signup":
                    if arr[1] == username:
                        details = arr[3]+","+arr[4]+","+arr[5]
                        break
        if output != "Insufficient Balance is Wallet":
            today = date.today()            
            data = "bookorder#"+pid+"#"+username+"#"+details+"#"+str(today)+"\n"
            saveDataBlockChain(data,"addcart")
            data = username+"#"+amount+"#"+str(today)+"#Paid Towards "+pid+" purchased\n"
            saveDataBlockChain(data,"addwallet")
        context= {'data':output}
        return render(request, 'UserScreen.html', context)      

def SearchProductAction(request):
    if request.method == 'POST':
        ptype = request.POST.get('t1', False)
        output = '<table border=1 align=center>'
        output+='<tr><th><font size=3 color=black>Service Provider Name</font></th>'
        output+='<th><font size=3 color=black>Product Name</font></th>'
        output+='<th><font size=3 color=black>Price</font></th>'
        output+='<th><font size=3 color=black>Quantity</font></th>'
        output+='<th><font size=3 color=black>Description</font></th>'
        output+='<th><font size=3 color=black>Image</font></th>'
        output+='<th><font size=3 color=black>Purchase Product</font></th></tr>'
        readDetails("addproduct")
        rows = details.split("\n")
        for i in range(len(rows)-1):
            arr = rows[i].split("#")
            print("my=== "+str(arr[0])+" "+arr[1]+" "+arr[2]+" "+ptype)
            if arr[0] == 'addproduct':
                if arr[2] == ptype:
                    output+='<tr><td><font size=3 color=black>'+arr[1]+'</font></td>'
                    output+='<td><font size=3 color=black>'+arr[2]+'</font></td>'
                    output+='<td><font size=3 color=black>'+str(arr[3])+'</font></td>'
                    output+='<td><font size=3 color=black>'+str(arr[4])+'</font></td>'
                    output+='<td><font size=3 color=black>'+arr[5]+'</font></td>'
                    #content = api.get_pyobj(arr[6])
                    #content = pickle.loads(content)
                    '''
                    if os.path.exists('FinancialApp/static/product.png'):
                        os.remove('FinancialApp/static/product.png')
                    with open('FinancialApp/static/product.png', "wb") as file:
                        file.write(content)
                    file.close()
                    '''
                    img_path = "static/Products/"+arr[6]
                    print(img_path+"======")
                    output+='<td><img src="'+img_path+'" width="200" height="200"></img></td>'
                    output+='<td><a href=\'BookOrders?farmer='+arr[1]+'&crop='+arr[2]+'&amount='+str(arr[3])+'\'><font size=3 color=black>Click Here</font></a></td></tr>'
        output+="</table><br/><br/><br/><br/><br/><br/>"
        context= {'data':output}
        return render(request, 'SearchProducts.html', context)              
        
    
def AddProductAction(request):
    if request.method == 'POST':
        cname = request.POST.get('t1', False)
        qty = request.POST.get('t2', False)
        price = request.POST.get('t3', False)
        desc = request.POST.get('t4', False)
        image = request.FILES['t5']
        imagename = request.FILES['t5'].name
        user = ''
        with open("session.txt", "r") as file:
            for line in file:
                user = line.strip('\n')
        file.close()
        '''
        myfile = pickle.dumps(image)
        hashcode = api.add_pyobj(myfile)
        '''
        hashcode = imagename
        fs = FileSystemStorage()
        filename = fs.save('FinancialApp/static/Products/'+imagename, image)
        data = "addproduct#"+user+"#"+cname+"#"+price+"#"+qty+"#"+desc+"#"+hashcode+"\n"
        saveDataBlockChain(data,"addproduct")
        context= {'data':"Product details saved and IPFS image storage hashcode = "+hashcode}
        return render(request, 'AddProduct.html', context)
        
   
def Signup(request):
    if request.method == 'POST':
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        contact = request.POST.get('contact', False)
        email = request.POST.get('email', False)
        address = request.POST.get('address', False)
        usertype = request.POST.get('type', False)
        record = 'none'
        readDetails("adduser")
        rows = details.split("\n")
        for i in range(len(rows)-1):
            arr = rows[i].split("#")
            if arr[0] == "signup":
                if arr[1] == username:
                    record = "exists"
                    break
        if record == 'none':
            data = "signup#"+username+"#"+password+"#"+contact+"#"+email+"#"+address+"#"+usertype+"\n"
            saveDataBlockChain(data,"adduser")
            context= {'data':'Signup process completd and record saved in Blockchain'}
            return render(request, 'Register.html', context)
        else:
            context= {'data':username+'Username already exists'}
            return render(request, 'Register.html', context)    



def UserLogin(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        usertype = request.POST.get('type', False)
        status = 'none'
        readDetails("adduser")
        rows = details.split("\n")
        for i in range(len(rows)-1):
            arr = rows[i].split("#")
            if arr[0] == "signup":
                if arr[1] == username and arr[2] == password and arr[6] == usertype:
                    status = 'success'
                    break
        if status == 'success' and usertype == 'Service Provider':
            file = open('session.txt','w')
            file.write(username)
            file.close()
            context= {'data':"Welcome "+username}
            return render(request, 'ServiceProviderScreen.html', context)
        elif status == 'success' and usertype == 'User':
            file = open('session.txt','w')
            file.write(username)
            file.close()
            context= {'data':"Welcome "+username}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'Invalid login details'}
            return render(request, 'Login.html', context)            


        
        
def predict(request):
    # import libraries
    import math
    import pandas_ta as ta
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, Dropout
    import matplotlib.pyplot as plt
    from datetime import datetime as dt
    #from binance import Client
    from binance.client import Client
    import csv

# ignore warnings
    import warnings
    warnings.filterwarnings("ignore")
    # create client object from binance library
    client = Client(None, None)     
    # defining the function that fetch the historical data of the wanted coin
    def getdata(symbol, period, start, end):
        """
        This function gives us the historical candle data of the desired coin
    :param symbol: coin name that you want to get data
    :param period: time period of candles
    :param start: starting date of candles data
    :param end: end date of candles data
    :return: candles data
        """
        candles = client.get_historical_klines(symbol, period, start, end)
        return candles
    def make_csv(symbol, candles):
        """
    This function makes csv file of desired coin with defined properties
    :param symbol: coin name that you want to make csv file
    :param candles: historical data of the desired coin
    :return: historical data in csv file
        """
        csvfile = open(symbol + ".csv", "a", newline="")
        cursor = csv.writer(csvfile)
        for i in candles:
            cursor.writerow(i)
        csvfile.close()
    # defining the function that make csv files of the historical data of the wanted multiple coin
    def multiple_csv(symbols, interval, start, end):
        """
    This function makes csv file for each coin in symbols parameter with defined properties
    :param symbols: list of multiple coin names that you want to make csv file
    :param interval: time period of candles (default: client.KLINE_INTERVAL_1DAY you can change the interval)
    :param start: starting date of candles data
    :param end: end date of candles data
    :return: historical data of multiple coin in csv files
        """
        for i in symbols:
            make_csv(i, getdata(i, interval, str(start), str(end)))
            print(i, "csv file is ready.")
    # defining function the that turn the timestamp to the date 
    def calculate_time(timestamp):
        """
    This function turns the timestamp to the date
    :param timestamp: given timestamp
    :return: date according to given timestamp
        """
        return dt.fromtimestamp(timestamp/1000)
    # get Bitcoin historical data and make csv
    multiple_csv(["BTCUSDT"], client.KLINE_INTERVAL_1DAY, "8 November 2010", "20 September 2022")
    # read Bitcoin histroical data as dataframe with column names
    headers = ["Open Time", "Open", "High", "Low", "Close", "Volume", "Close Time", "QAV", "NAT", "TBBAV", "TBQAV", "Ignore"]
    data = pd.read_csv("BTCUSDT.csv", names=headers)
    print(data.head())
    # Turn "Open Time" and "Close Time" columns to Date
    open_date = []
    for i in data["Open Time"]:
        open_date.append(calculate_time(i))
    data["Open Time"] = open_date

    close_date = []
    for i in data["Close Time"]:
        close_date.append(calculate_time(i))
    data["Close Time"] = close_date
    print(data)
    # Visualize the close price history
    plt.figure(figsize=(16, 8))
    plt.title("Bitcoin Price History")
    plt.plot(data["Close Time"], data["Close"])
    plt.xlabel("Time", fontsize=14,)
    plt.ylabel("USDT", fontsize=14)
    plt.show()
    # Create new data with only the "Close" column
    close = data.filter(["Close"])
# Convert the dataframe to a np array
    close_array = close.values
# See the train data len
    train_close_len = math.ceil(len(close_array) * 0.8)
    print(train_close_len)
    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_array)
    print(scaled_data)
    # Create the training dataset
    train_data = scaled_data[0 : train_close_len, :]
# Create X_train and y_train
    X_train = []
    y_train = []
    for i in range(60, len(train_data)):
        X_train.append(train_data[i - 60 : i, 0])
        y_train.append(train_data[i, 0])
        if i <= 60:
            print(X_train)
            print(y_train)
    #  make X_train and y_train np array
    X_train, y_train = np.array(X_train), np.array(y_train)
    # reshape the data
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    print(X_train.shape)
    # create the testing dataset
    test_data = scaled_data[train_close_len - 60 : , :]
# create X_test and y_test
    X_test = []
    y_test = data.iloc[train_close_len : , :]
    for i in range(60, len(test_data)):
        X_test.append(test_data[i - 60 : i, 0])
    # convert the test data to a np array and reshape the test data
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    # Build the LSTM Model
    model = Sequential()

    model.add(LSTM(units=512, return_sequences=True, activation='relu', input_shape=(X_train.shape[1], 1)))


    model.add(LSTM(units=256, activation='relu', return_sequences=False))


    model.add(Dense(units=1))
    # compile the LSTM model
    model.compile(optimizer="Adam", loss="mean_squared_error", metrics=['mae'])
    # train the LSTM model
    model.fit(X_train, y_train,
          epochs=3,
          batch_size=100, 
          verbose=1)
    # predict with LSTM model
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    # plot the data
    train = close[:train_close_len]
    valid = close[train_close_len:]
    valid["Predictions"] = predictions
#visualize the data
    plt.figure(figsize=(16, 8))
    plt.title("LSTM Model")
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("USDT", fontsize=14)
    plt.plot(data["Close Time"][:train_close_len], train["Close"])
    plt.plot(data["Close Time"][train_close_len:], valid[["Close", "Predictions"]])
    plt.legend(["Train", "Validation", "Predictions"], loc="lower right")
    plt.show()
    # change the parameters of first LSTM model and build the Optimized LSTM Model
    optimized_model = Sequential()

    optimized_model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))

    optimized_model.add(LSTM(100, return_sequences=False))

    optimized_model.add(Dense(50))

    optimized_model.add(Dense(1))
    # compile the model
    optimized_model.compile(optimizer="Adam", loss="mean_squared_error", metrics=['mae'])
    # train the optimized model
    optimized_model.fit(X_train, y_train, 
          batch_size=10, 
          epochs=3, 
          verbose=1)
    # Predict with optimized LSTM model
    o_predictions = optimized_model.predict(X_test)
    o_predictions = scaler.inverse_transform(o_predictions)
    # plot the data
    train = close[:train_close_len]
    valid = close[train_close_len:]
    valid["Predictions"] = o_predictions
#visualize the data
    plt.figure(figsize=(16, 8))
    plt.title("Optimized LSTM Model")
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("USDT", fontsize=14)
    plt.plot(data["Close Time"][:train_close_len], train["Close"])
    plt.plot(data["Close Time"][train_close_len:], valid[["Close", "Predictions"]])
    plt.legend(["Train", "Validation", "Predictions"], loc="lower right")
    plt.show()
    
    return render(request,"predict.html")


        
            
