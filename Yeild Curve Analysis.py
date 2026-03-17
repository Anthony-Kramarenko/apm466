import scipy.optimize as sc
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as pl



#_________________________________|input_DataFrame|____________________________________________________________________
#Input:
#Candiand Bond Data.csv

Bond_DataFrame = pd.read_csv('Canadian Bond Data.csv')

#______________________________________________________________________________________________________________________


#______________________________________________________________________________________________________________________

#                                 Yeild Curve, Spot Curve And One year Forward Curve Classes

#______________________________________________________________________________________________________________________

def subtract_6_months(date):
    month = date.month - 6
    year = date.year

    if month <= 0:
        month += 12
        year -= 1

    return dt.datetime(year, month, date.day)

class Bond():
    def __init__(self, ISIN, Issue_Date, Coupon, Maturity_Date, Face_Value):
        self.ISIN = ISIN
        self.Issue_Date = Issue_Date
        self.Coupon = Coupon
        self.Maturity_Date = Maturity_Date
        self.Face_Value= Face_Value
        self.Issue_date = Issue_Date


class Bond_Analytics(Bond):
    def __init__(self, ISIN, Issue_Date, Coupon, Maturity_Date, Face_Value,Current_Date, Close_Price):
        super().__init__( ISIN, Issue_Date, Coupon, Maturity_Date, Face_Value)
        self.Current_Date = Current_Date
        self.Close_Price = Close_Price

    def payments_due(self):

        if self.Current_Date > self.Maturity_Date:
            return [], []

        payment_schedule = [self.Maturity_Date]
        payment = [0.5*self.Coupon*self.Face_Value + self.Face_Value]

        temp_date = subtract_6_months(self.Maturity_Date)

        while temp_date > self.Current_Date:
            payment_schedule.insert(0, temp_date)
            payment.insert(0,self.Coupon*0.5*self.Face_Value)

            temp_date = subtract_6_months(temp_date)

        return payment, payment_schedule
    def last_Coupon_Date(self):
        Possibly_Last_Pmt = subtract_6_months(self.payments_due()[1][0])
        if self.Issue_Date > Possibly_Last_Pmt:
            return self.Issue_date
        return Possibly_Last_Pmt

    def Dirty_Price(self):
        return self.Close_Price+0.5*self.Coupon*self.Face_Value*(self.Current_Date-self.last_Coupon_Date()).days/182.5

    def PV(self, Yield):
        Payment, Payment_Schedule = self.payments_due()
        Pv = 0

        for i in range(len(Payment)):
            t =(Payment_Schedule[i] -self.Current_Date).days/365.25
            Pv += Payment[i]/(1 + Yield*0.5)**(2*t)
        return Pv
    def Yield(self):
        return sc.newton(lambda Yield: self.PV(Yield)- self.Dirty_Price(), x0=0.02, maxiter=1500)

class Yield_Curve():
    def __init__(self, YTM_Dates=[], YTM = []):
        self.YTM_Dates = YTM_Dates
        self.YTM = YTM

    def Insert_Bond(self, New_Bond):
        self.YTM.append(New_Bond.Yield())
        self.YTM_Dates.append(New_Bond.Maturity_Date)


    def Linear_interpolation(self, Current_Date,max_years):
        '''
        return time to maturity with linearly interpolated yeilds from YTM
        '''
        target_tenors = [x * 0.5 for x in range(int(max_years * 2) + 1)]


        known_tenors = [(d - Current_Date).days / 365.25 for d in self.YTM_Dates]
        known_yields = self.YTM

        interpolated_ytms = np.interp(target_tenors, known_tenors, known_yields)

        return target_tenors,list(interpolated_ytms)

class Spot_curve():
    def __init__(self,  Spot_Rates_Dates= [], Spot_Rates= []):
        self.Spot_Rates_Dates = Spot_Rates_Dates
        self.Spot_Rates = Spot_Rates

    def Bootstrap_Insert(self, New_Bond):
        if self.Spot_Rates == []:
            self.Spot_Rates_Dates.append(New_Bond.Maturity_Date)
            spot = New_Bond.Yield()
            self.Spot_Rates.append(spot)
            return spot
        if subtract_6_months(New_Bond.Maturity_Date) !=  self.Spot_Rates_Dates[-1]:
            raise TypeError("Bond input is not compatible for bootstrapping")
            return

        price = New_Bond.Dirty_Price()
        face = New_Bond.Face_Value
        coupon_rate = New_Bond.Coupon
        coupon = face * coupon_rate / 2

        pv_known = 0
        for i, r in enumerate(self.Spot_Rates):
            t = (i + 1) * 0.5
            pv_known += coupon / (1 + r / 2) ** (2 * t)

        n = len(self.Spot_Rates) + 1
        t_n = n * 0.5

        pv_remaining = price - pv_known
        if pv_remaining <= 0:
            raise ValueError("Price too low to bootstrap new spot rate")

        new_spot = 2 * (((coupon + face) / pv_remaining) ** (1 / (2 * t_n)) - 1)
        self.Spot_Rates.append(new_spot)
        self.Spot_Rates_Dates.append(New_Bond.Maturity_Date)

        return new_spot
    def Linear_interpolation(self, Current_Date, max_years):
        '''
        return time to maturity with linearly interpolated spots
        '''
        target_tenors = [x * 0.5 for x in range(int(max_years * 2) + 1)]

        known_tenors = [(d - Current_Date).days / 365.25 for d in self.Spot_Rates_Dates]
        known_spots = self.Spot_Rates

        interpolated_spots = np.interp(target_tenors, known_tenors, known_spots)

        return target_tenors, list(interpolated_spots)


class y1_Forward_Curve():

    def __init__(self):
        self.Forward_Rates = []
        self.Forward_years = []

    def Build_Forward_Curve(self, year, Spot_Rates):

        self.Forward_Rates = []
        self.Forward_years = []


        for i in range(len(Spot_Rates) - 1):
            s_t = Spot_Rates[i]
            s_t1 = Spot_Rates[i + 1]

            t = year[i]
            t1 = year[i + 1]


            numerator = (1 + s_t1) ** t1
            denominator = (1 + s_t) ** t

            fwd_rate = (numerator / denominator) ** (1 / (t1 - t)) - 1

            self.Forward_Rates.append(fwd_rate)

            self.Forward_years.append(t)

        return self.Forward_years, self.Forward_Rates
#_________________________________________________________________________________________________________________________________

#                                            Data Cleaning

#_____________________________________________________________________________________________________________________________



# Make a copy of the DataFrame for modifications
Bond_DF_Copy = Bond_DataFrame.copy()

#Convert all date columns into Datetime variables

Bond_DF_Copy["Issue date"] = pd.to_datetime(Bond_DF_Copy["Issue date"], format='%m/%d/%Y')
Bond_DF_Copy["Maturity Date"] = pd.to_datetime(Bond_DF_Copy["Maturity Date"], format='%m/%d/%Y')

#Clean Coupon Rates:

Bond_DF_Copy["Coupon"] = Bond_DF_Copy["Coupon"].str.rstrip('%').astype(float) * 0.01

#Remove bonds that will not be Used during Boot Strapping
Bond_DF_Copy = Bond_DF_Copy[
    ((Bond_DF_Copy['Maturity Date'].dt.month == 3) & (Bond_DF_Copy['Maturity Date'].dt.day == 1)) |
    ((Bond_DF_Copy['Maturity Date'].dt.month == 9) & (Bond_DF_Copy['Maturity Date'].dt.day == 1))
].reset_index(drop=True)

#Create new col For Bond name 'CAN' + 'Coupon in present' + "Maturity Month (3 letter short form)" + "year of maturity "

Bond_Name = (
    'CAN' +
    (Bond_DF_Copy['Coupon']*100).round(2).astype(str) +        # Coupon in percent
    Bond_DF_Copy['Maturity Date'].dt.strftime('%b') +            # 3-letter month abbreviation
    Bond_DF_Copy['Maturity Date'].dt.strftime('%y')             # Year
)
Bond_DF_Copy.insert(0, "Bond Name", Bond_Name)

# Sort rows by Maturities:
Bond_DF_Copy = Bond_DF_Copy.sort_values(by="Maturity Date").reset_index(drop=True)

#As building 5 year curve, I will remove data that matures past 2029
Bond_DF_Copy = Bond_DF_Copy[Bond_DF_Copy["Maturity Date"].dt.year <= 2029]

#colect all dates for which Yeild curves build
close_cols = [col for col in Bond_DF_Copy.columns if col.startswith("Close")]

#______________________________________________________________________________________________________________________

#                                      Plotting Curves Yeild Curve (interpolated)

#______________________________________________________________________________________________________________________


YTM_matrix = []
#Build Yield curves and plot
pl.title('Yield Curves')
for Close_Date in close_cols[:]:
    d = pd.to_datetime(Close_Date.split()[-1], format="%m-%d-%Y")
    yield_curve  = Yield_Curve([],[])
    for index, bond in Bond_DF_Copy.iterrows():
        #collect data for yeild curve
        _bond = Bond_Analytics(Bond_DF_Copy.loc[index,'ISIN'],
                       Bond_DF_Copy.loc[index, 'Issue date'],
                       Bond_DF_Copy.loc[index, 'Coupon'],
                       Bond_DF_Copy.loc[index,"Maturity Date"],
                        100,
                       d,
                       Bond_DF_Copy.loc[index, Close_Date])

        #build yield curve
        yield_curve.Insert_Bond(_bond)

    xy=yield_curve.Linear_interpolation(d, 5)
    pl.plot(xy[0],xy[1],'o-', label= Close_Date )  # Plot the curve
    YTM_matrix.append(xy[1])


pl.legend()
pl.grid(True)
pl.xlabel("Time to Maturity (Years)")
pl.ylabel("Yield to Maturity")
pl.show()

#______________________________________________________________________________________________________________________

#                                            Principal Component Analysis

#_______________________________________________________________________________________________________________________


X = np.array(YTM_matrix)
log_returns = np.diff(np.log(X), axis=0)
cov_matrix = np.cov(log_returns, rowvar=False)

#print Covariance Martix

print(f"The Covariance Matrix for Yield Curves{cov_matrix}")

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

#. Sort by Eigenvalues (NumPy does not return them sorted by default)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Print Results
print("--- Principal Component Analysis (PCA) Results ---\n")
total_variance = np.sum(eigenvalues)
for i in range(5):
    explained_var = (eigenvalues[i] / total_variance) * 100
    print(f"Factor {i+1} (PC{i+1}):")
    print(f"  - Eigen Values: {eigenvalues[i]}")
    print(f"  - Explained Variance: {explained_var:.2f}%")
    print(f"  - Eigenvector (Loadings): {eigenvectors[:, i].round(3)}")
    print("-" * 40)


#______________________________________________________________________________________________________________________



#                                  Spot rate curve, interpolated



#_______________________________________________________________________________________________________________________

#Building Matrix for PCA
Interpolated_Spot_Matrix = []


#Build Yield curves and plot
pl.title('Spot Rate Curves')
for Close_Date in close_cols[:]:
    d = pd.to_datetime(Close_Date.split()[-1], format="%m-%d-%Y")
    spot_curve  = Spot_curve([],[])
    for index, bond in Bond_DF_Copy.iterrows():
        #collect data for yield curve
        _bond = Bond_Analytics(Bond_DF_Copy.loc[index,'ISIN'],
                       Bond_DF_Copy.loc[index, 'Issue date'],
                       Bond_DF_Copy.loc[index, 'Coupon'],
                       Bond_DF_Copy.loc[index,"Maturity Date"],
                        100,
                       d,
                       Bond_DF_Copy.loc[index, Close_Date])

        #build yield curve
        spot_curve.Bootstrap_Insert(_bond)

    xy=spot_curve.Linear_interpolation(d, 5)

    # for one year forward rates extract interpolated Spot information:

    Interpolated_Spot_Matrix.append(xy)


    pl.plot(xy[0],xy[1],'o-', label= Close_Date )  # Plot the curve


pl.legend()
pl.grid(True)
pl.xlabel("Time to Maturity (Years)")
pl.ylabel("Spot Rate")
pl.show()


#_____________________________________________________________________________________________________________________

#                                                Forward Interpolated Rates

#___________________________________________________________________________________________________________________

pl.title('Forward Rate Curves ')
#for PCA
Forward_matrix = []
for I_S_Curve in Interpolated_Spot_Matrix:
    forward_curve = y1_Forward_Curve()
    forward_curve.Build_Forward_Curve(I_S_Curve[0],I_S_Curve[1])
    pl.plot(forward_curve.Forward_years[2::2], forward_curve.Forward_Rates[2::2], 'o-', label=Close_Date)  # Plot the curve
    Forward_matrix.append(forward_curve.Forward_Rates[2::2])

pl.legend()
pl.grid(True)
pl.xlabel("Forward Years")
pl.ylabel("One-Year Forward Rate")
pl.show()
#______________________________________________________________________________________________________________________

#                                    PCA for One Year Forward rates

#_______________________________________________________________________________________________________________________

X = np.array(Forward_matrix)
log_returns = np.diff(np.log(X), axis=0)
cov_matrix = np.cov(log_returns, rowvar=False)

#print Covariance Martix

print(f"The Covariance Matrix for Forward Rate Curves{cov_matrix}")

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort by Eigenvalues
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Print Results
print("--- Principal Component Analysis (PCA) Results ---\n")
total_variance = np.sum(eigenvalues)
for i in range(4):
    explained_var = (eigenvalues[i] / total_variance) * 100
    print(f"Factor {i+1} (PC{i+1}):")
    print(f"  - Eigen Values: {eigenvalues[i]}")
    print(f"  - Explained Variance: {explained_var:.2f}%")
    print(f"  - Eigenvector (Loadings): {eigenvectors[:, i].round(3)}")
    print("-" * 40)