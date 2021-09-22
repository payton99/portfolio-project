import requests
import pandas as pd
import numpy as np
from datetime import date
from csv import writer
import concurrent.futures
from config import *



class Portfolio():

    '''Create a portfolio summary based on list of stocks provided'''

    def __init__(self, df, my_stocks):
        self.df = df
        self.my_stocks = my_stocks
        


    def holdings(self):
        agg_quant = self.df.groupby(['Symbol']).sum(['Quantity']).round(4)

        # Avoid SettingWithCopyWarning by including .copy() 
        self.current_holdings = agg_quant.loc[agg_quant.Quantity > 0].copy()
        self.current_holdings.drop(['Price', 'Amount'], axis = 1, inplace = True)
        return self.current_holdings
    

    def cost_basis(self):
        self.holdings()

        '''
        Use FIFO accounting method (First in, first out)
        Example using VOO:

        Bought 2 shares @ 246.76 = $498.47
        Bought 2 shares @ 274.59 = $549.18
        Sold 2 shares @ 296.40
          -But the selling price is not 296.40, but
          -rather it is the original price of 246.76

        Sold 1 share
          -Now we sell at the next earliest buy price,
          -in this case 274.59

        Now our cost basis ledger looks like this:
            -> 1 share @ 274.59

        Next, we add in ONLY reinvested dividends:
                                                     ---------
        274.59 + 2.87 + 2.63 + 2.79 + 1.30 + 1.37 = | $285.55 |
                                                     ---------
        '''

        # loop through, add all 'buy's
        #
        # if there are any 'sell's, subtract the number
        # of them from the number of 'buy's
        #
        # then add any reinvested dividends

        self.df = self.df.sort_values(by = ['Symbol', 'Date'])
        self.df['changed'] = self.df['Symbol'].ne(self.df['Symbol'].shift().bfill()).astype(int)
        self.df.dropna(subset = ['Symbol'], inplace = True)

        df_num = self.df.to_numpy()

        divs = 0
        buffer = ''
        price, quant, amount = [], [], []

        for row in df_num:
            if row[3] not in my_stocks:
                continue
            
            if row[-1] == 1:
                if len(quant) == 0:
                    price.append(row[4])
                    quant.append(row[2])
                else:
                    buffer = 'set'
            
            else:
                if row[1] == 'Bought':
                    price.append(row[4])
                    quant.append(row[2])
                
                elif row[1] == 'Reinvestment':
                    divs = divs + row[5]
                
                elif row[1] == 'Canceled':
                    price.append((-1 * row[4]))
                    quant.append((-1 * row[2]))
                
                if len(quant) != 0:
                    if quant[0] != 0:
                        if row[1] == 'Sold':
                            quant[0] = quant[0] + row[2]
                            if quant[0] == 0:
                                quant.pop(0)               
                                price.pop(0)
                
    
            if buffer == 'set':
                value = np.multiply(price, quant)
                sum_val = sum(value)
                sum_val -= divs

                amount.append(round((sum_val),2))

                price.clear()
                quant.clear()

                price.append(row[4])
                quant.append(row[2])

                buffer = ''
                divs = 0


        value = np.multiply(price, quant)
        sum_val = sum(value)
        sum_val -= divs
        amount.append(round((sum_val),2))
        
        self.current_holdings['Amount Paid'] = amount
        print(self.current_holdings)

    

    def _get_price(self, stock):
        quotes = requests.get(f'https://financialmodelingprep.com/api/v3/quote/{stock}?apikey={api_key}').json()
        return quotes[0]['price']


    def threading_price(self):

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            self.results = list(executor.map(self._get_price, self.my_stocks))
        
        return self.results
    

    def added_cols(self):
        self.cost_basis()
        self.threading_price()
        
                
        num_shares = [share for share in self.current_holdings['Quantity']]
        
        self.current_holdings['Price'] = self.results
        self.market_value = [round((self.results[i] * num_shares[i]),2) for i in range(len(self.results))]
        self.current_holdings['Market Value'] = self.market_value
        self.current_holdings['Profit/Loss'] = self.current_holdings['Market Value'] - self.current_holdings['Amount Paid']


        print(self.current_holdings)

    
    def industries(self):
        self.added_cols()
        industries = {}
        for stock in self.my_stocks:
            info = requests.get(f'https://financialmodelingprep.com/api/v3/profile/{stock}?apikey={api_key}').json()
            for ele in info:
                if ele['industry'] == '':
                    industries[ele['symbol']] = 'Fund'
                else:
                    industries[ele['symbol']] = ele['industry']
        
        d_frame = pd.DataFrame(industries.items(), columns=['Symbol', 'Industry'])
        d_frame['Market Value'] = self.market_value
        sum_industries = d_frame.groupby(['Industry']).sum(['Market Value'])
        print(sum_industries)
    


    def sectors(self):
        self.added_cols()
        sectors = {}
        for stock in self.my_stocks:
            info = requests.get(f'https://financialmodelingprep.com/api/v3/profile/{stock}?apikey={api_key}').json()
            for ele in info:
                if ele['sector'] == '':
                    sectors[ele['symbol']] = 'Fund'
                else:
                    sectors[ele['symbol']] = ele['sector']
        
        d_frame = pd.DataFrame(sectors.items(), columns=['Symbol', 'Sector'])
        d_frame['Market Value'] = self.market_value
        sum_sectors = d_frame.groupby(['Sector']).sum(['Market Value'])
        print(sum_sectors)


    def total_dividends(self):
        self.divs = self.df[self.df['Activity'] == 'Dividend']
        self.divs = self.divs.groupby('Symbol').sum('Amount').copy()
        self.divs.drop(['Price', 'Quantity'], axis = 1, inplace = True)
        print(self.divs)

    
    ## DCF Analysis function
    def dcf_analysis(self):
        self.threading_price()

        dcf = []
        for stock in self.my_stocks:
            quotes = requests.get(f'https://financialmodelingprep.com/api/v3/discounted-cash-flow/{stock}?apikey={api_key}')
            dcf.append(quotes.json())

        dcf_values = []
        for list in dcf:
            for dic in list:
                try:
                    dcf_values.append(round((dic['dcf']),2))
                except KeyError:
                    dcf_values.append(np.nan)
        
        dic = {'Symbol': self.my_stocks, 'Price': self.prices, 'DCF Analysis': dcf_values}
        self.dcf_data = pd.DataFrame(data = dic)
        over_under =  round(((self.dcf_data['DCF Analysis'] - self.dcf_data['Price']) / self.dcf_data['DCF Analysis']), 3)
        over_under = over_under * 100
        self.dcf_data['Over/Under'] = over_under

        return print(self.dcf_data)

    # Calculate total cost of portfolio
    def total_cost(self):
        costs = self.cost_basis()

        return print(round(sum(costs['Amount Paid']), 2))

    # Calculate portfolio total market value
    def total_value(self):
        cols = self.added_cols()

        print(round(sum(cols['Market Value']), 2))
    
    # Calculate total profit or loss
    def profit_loss(self):
        cols = self.added_cols()

        return print(round(sum(cols['Profit/Loss']), 2))
    

    ## Function to determine percent change of entire portfolio compared to S&P500
    ## on any given day.
    def daily_pct_change(self, startdate):
        df = self.df.sort_values(by = ['Symbol', 'Date'])
        df.dropna(subset = ['Symbol'], inplace = True)

        if (startdate > str(date.today())) | (startdate < str(df['Date'].min())):
            return print('Error: date not in range.')
        
        data = {}
        for index, row in df.iterrows():

            while str(row['Date']) < startdate:
                if row['Activity'] == 'Bought':
                    data[row['Symbol']] = row['Quantity']

                elif row['Activity'] == 'Reinvestment':
                    data[row['Symbol']] += row['Quantity']

                elif row['Activity'] == 'Canceled':
                    data[row['Symbol']] += row['Quantity']
                    
                break

        enddate = startdate

        open, close = [], []
        for stock in self.my_stocks:
            if stock in data:
                try:
                    quotes = requests.get(
                        f'https://financialmodelingprep.com/api/v3/historical-price-full/{stock}?from={startdate}&to={enddate}&apikey={api_key}').json()

                    open.append(quotes['historical'][0]['open'])
                    close.append(quotes['historical'][0]['close'])
            
                except KeyError:
                    open.append('NA')
                    close.append('NA')
                    continue
        try:
            total_open = [i * j for i, j in zip(open, data.values())]
            total_close = [i * j for i, j in zip(close, data.values())]

            pct_change = ((sum(total_close) - sum(total_open)) / sum(total_open))
            
            print('My Daily Percent Change: {}%'.format(round((100 * pct_change), 3)))

            sp500 = requests.get(f'https://financialmodelingprep.com/api/v3/historical-price-full/^GSPC?from={startdate}&to={enddate}&apikey={api_key}').json()
            sp_open = sp500['historical'][0]['open']
            sp_close = sp500['historical'][0]['close']
            sp_change = ((sp_close - sp_open) / sp_open)
            print('\nS&P 500 Daily Percent Change: {}%'.format(round((100 * sp_change), 3)))

        except TypeError:
            print('\nError: Markets not open this day.\n')



    ## Function to append rows to the csv file.
    def update_csv(self, file_name, contents):

        '''
        Portfolio CSV styling (all in string format):

        +--------------------------------------------------------+
        | Date | Activity | Quantity | Symbol | Price | $ Amount |
        +--------------------------------------------------------+

        'Bought' Ex:
        
        +-----------------------------------------------------------------+
        | '8/31/2021' | 'Bought' | '1' | 'MSFT' | '$245.60' | '($245.60)' |
        +-----------------------------------------------------------------+

        - 'contents' parameter is list
        - To update csv using this method:

        object.update_csv('file_name.csv', ['8/31/2021', 'Dividend', '0', 'MSFT', '$0.00', '$1.90'])

        '''

        with open(file_name, 'a') as f_object:
            write_obj = writer(f_object)
            write_obj.writerow(contents)


user1 = Portfolio(df, my_stocks)
# print(user1)

#### Test functions ####
# print(user1.holdings())
user1.cost_basis()
# user1.threading()
# print(user1.added_cols())
# user1.industries()
# user1.sectors()
# user1.total_dividends()
# user1.dcf_analysis()
# user1.profit_loss()
# user1.daily_pct_change('2021-09-20')

# user1.update_csv('portfolio-history2.csv', ['9/21/2021', 'Bought', '1', 'VZ', '$54.13', '($54.13)'])



