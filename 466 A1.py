

import matplotlib.pyplot as pl
import scipy.optimize as sc
import pandas as pd
import numpy as np
from pandas import DataFrame
import math
from scipy.optimize import newton


##################################################################dirty price helper functions##########################
def dprice(t, coupon, price):
    return (((127 +(t-6))/365)* coupon + price)
def day_count (days):
    i = 0
    day_counter = 0
    temp_str = ''
    while days[i] != '/':
        temp_str += days[i]
        i += 1
    day_counter += monthtoday(int(temp_str))
    temp_str = ''
    i += 1
    while days[i] != '/':
        temp_str += days[i]
        i += 1
    day_counter += int(temp_str)
    i += 1
    temp_str = ''
    #print("day counter:")
    #print(day_counter)
    while i < len(days):
        temp_str += days[i]
        i += 1
    day_counter += 365*int(temp_str)
    #print(day_counter)
    return day_counter

##############################################################ytm helper functions##################################
def monthtoday(month):
    lof = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    sol = 0
    month -= 2
    while month >= 0:
        sol += lof[month]
        month -= 1
    return sol
days = [6,7,8,9,10,13,14,15,16,17]
# computes
def numdaysfromindex(T, i):
    day = [6,7,8,9,10,13,14,15,16,17]
    days = day_count(T) - day[i] - 365*2025
    return 2*(days/365)

def payments_annu(T,i):
    output = []
    days_remaining= 0.5*numdaysfromindex(T, i)#annualized
    while days_remaining > 0:
        output.append( days_remaining)
        days_remaining -= 0.5
    output.reverse()
    #print("output:")
    #print(output)
    return output

def payments_semi(T,i):
    output = []
    days_remaining= numdaysfromindex(T, i)#annualized
    while days_remaining > 0:
        output.append( days_remaining)
        days_remaining -= 1
    output.reverse()
    #print("output:")
    #print(output)
    return output


def PV(i, T, r , coupon_str):
    #print("star")
    coupon_pmt = float(coupon_str[:-1])*0.5
    bond = 0
    for j in range(0, len(payments_semi(T,i))):
        if j == (len(payments_semi(T,i)) - 1) :
            bond += (coupon_pmt +100)/((1+r )**payments_semi(T,i)[j])
        else:
            bond += coupon_pmt/((1+r )**payments_semi(T,i)[j])
        #print(bond)
    #print("end")
    return bond

def YTM(i,T, coupon_str, dirty_price):

    coupon_pmt = float(coupon_str[:-1]) * 0.5
    ntn = sc.newton(lambda r: PV(i, T, r, coupon_str) - dirty_price, x0=0.02, maxiter=1500)
    #print(ntn)
    return ntn*2

############################spot curve functions#########################################################
#this fnction computes the priveious coup rates
def previous_compunding(spots, coupon, T, t):
    payment = payments_annu(T,t)
    comp = 0
    cpn = float(coupon[:-1])*0.5
   # print(coupon)
    #print(len(payment))
    for i in range(0,len(spots)):
        comp += cpn/ (1 + spots[i])**payment[i]
    return comp
# this solves for the one unkon spot rate in th equation 
def solve_for_r(spots, coupon, T, t, dirty_price):
    payment = payments_annu(T,t)
    cpn_pmt = float(coupon[:-1])*0.5
    r_n = ((cpn_pmt +100)/(dirty_price - previous_compunding(spots, coupon, T, t)))**(1/payment[-1])-1
    return  r_n
####################################################################################################### contract

# this returns the average
def tw_average(aytm, bytm, adate, bdate, targetdate):
    weight_a= (day_count(targetdate)-day_count(adate))/(day_count(bdate)-day_count(adate))
    #print(weight_a)
    return weight_a*aytm + (1-weight_a)*bytm


###############################################################################extend
# get last 2 point extract line equation and add it on

#extract_line(avalue, bvalue, adate, bdate,target_date)
def extract_line(avalue, bvalue, adate, bdate,target_date):
    gradient= (bvalue - avalue)/(numdaysfromindex(bdate,9)*0.5 - 0.5*numdaysfromindex(adate,9))
    #print(gradient)
    intersept= bvalue-gradient*numdaysfromindex(bdate, 9)*0.5
    #print(intersept)
    #print (intersept)
    #print("intercept")
    #print(gradient)
    return intersept + gradient*numdaysfromindex(target_date,9)*0.5

#############################################################################################################################
# daily log returns
def dailylog(u,d):
    return  math.log(u/d)

####################################################################################################
Aone = pd.read_csv('C:/Users/avkra/OneDrive/Desktop/466/A1 446/Assignment 1 466.csv')
# choosing my 10 bonds:
# first we observe the avalible maturity dates that we have:
lister = list(Aone.loc[:,"Maturity Date"])

# next we observe which of these bonds is most appropriate for bootstrapping:

#selecting the ideal maturity bonds (3, 9 months)
Mat_of_ten_bds= []
for element in lister:
    if int(element[0])== 3 or int(element[0]) == 9:
        if 2030 > int(element[-4:]):
            if lister.count(element) ==1:
                Mat_of_ten_bds.append(element)

extract_col = list(Aone.columns)

#sort maturities in order:################################################################
def sum(date):
    return int(date[0]) + int(date[-4:])*365
sortedlist = []
while len(Mat_of_ten_bds) != 0:
    min = Mat_of_ten_bds[0]
    for element in Mat_of_ten_bds:
        if sum(element) < sum(min):
            min = element
    sortedlist.append(min)
    Mat_of_ten_bds.pop(Mat_of_ten_bds.index(min))
Mat_of_ten_bds = sortedlist
#build data frame for chosen bonds ####################################################
data={}
for names in list(Aone.columns):
    data[names] = []
df = pd.DataFrame(data)
for maturity in Mat_of_ten_bds:
    row = Aone[Aone['Maturity Date'] == maturity]
    df = pd.concat([df, row], ignore_index=True)

# To easily refer to each bond we will create names now ###########################################
# CAN 2.5 Jun 24
#

namelist = []
for i in range(0,10):
    if ( i % 2 == 0):
        namelist.append('CAN ' + str(df.iloc[i,2])[:-1] + ' Mar 1')
    if (i % 2 ==1 ):
        namelist.append('CAN ' + str(df.iloc[i, 2])[:-1] + ' Sep 1')
df['Name'] = namelist
#print(df)
#dirty price #################################################################################################
#days since last coupon payment: 127 from the 6th jan point:
data_dirty = {}
df_dirty = pd.DataFrame(data_dirty)
df_dirty["Name"] = namelist
df_dirty['Maturity Date ']=df['Maturity Date']
list_of_dates = [6,7,8,9,10,13,14,15,16,17]
for j in range(0,10):
    d  = []
    for i in range(0,10):
        d.append(dprice(list_of_dates[j], float(str(df.iloc[i,2])[:-1]), float(df.iloc[i,j +4 ]) ))
    df_dirty[ ("Dirty Price for Jan" +str(list_of_dates[j]))] = d
#print(df_dirty )

 # now we have compute yeild to maturity of each bond for every fixed date
#initialize the data frame
lt = ["Name",
                 'Maturity Date',
                 "Yield Curve for Jan 6 2025",
                 "Yield Curve for Jan 7 2025",
                 "Yield Curve for Jan 8 2025",
                 "Yield Curve for Jan 9 2025",
                 "Yield Curve for Jan 10 2025",
                 "Yield Curve for Jan 13 2025",
                 "Yield Curve for Jan 14 2025",
                 "Yield Curve for Jan 15 2025",
                 "Yield Curve for Jan 16 2025",
                 "Yield Curve for Jan 17 2025"]
#print(df.columns)
colmns_ytm_df = lt
#print(df.columns)
data1={}
df_YTM = pd.DataFrame(data1)
for element in colmns_ytm_df:
    df_YTM[element] = {}
df_YTM["Name"] = namelist
df_YTM['Maturity Date']=df['Maturity Date']
# now we populate with the required values
for i in range(0,10):
    for j in range(0,10):
        df_YTM.iloc[j,i + 2] = YTM(i,df.iloc[j,3], df.iloc[j,2], df_dirty.iloc[j,i +2])
Save = df_YTM
#################################################################################################
#Notice that YTM curves not full 5 year so we extend
# solving for 17/1/2030
df_lastYTM = pd.DataFrame({})
df_lastYTM['Maturity Date'] = ['1/17/2030']
df_lastYTM["Name"] = ['5 year YTM']


for i in range(0,10):
    df_lastYTM[colmns_ytm_df[i+2]] = [extract_line(df_YTM.iloc[8, i+2], df_YTM.iloc[9,i +2], '3/1/2029', '9/1/2029','1/17/2030')]
df_YTM = pd.concat([df_YTM,df_lastYTM], axis = 0, ignore_index= True )


#also these curves are at year zero so well shall fix that  aswell
df_firstYTM = pd.DataFrame({})
df_firstYTM['Maturity Date'] = ['1/17/2025']
df_firstYTM["Name"] = ['0 year spot']


for i in range(0,10):
    df_firstYTM[colmns_ytm_df[i+2]] = [extract_line(df_YTM.iloc[1, i+2], df_YTM.iloc[0,i +2], '9/1/2025', '3/1/2025','1/17/2025')]
df_YTM = pd.concat([df_firstYTM,df_YTM], axis = 0, ignore_index= True )



####################################################################################################################
# now we shall build the spot curve
# because of our strategic selection of bonds they all follow the same payment schedule
day = [6,7,8,9,10,13,14,15,16,17]
data_spot = {}
df_spot = pd.DataFrame(data_spot)

l = ["Name",
                 'Maturity Date',
                 "Spot Curve for Jan 6 2025",
                 "Spot Curve for Jan 7 2025",
                 "Spot Curve for Jan 8 2025",
                 "Spot Curve for Jan 9 2025",
                 "Spot Curve for Jan 10 2025",
                 "Spot Curve for Jan 13 2025",
                 "Spot Curve for Jan 14 2025",
                 "Spot Curve for Jan 15 2025",
                 "Spot Curve for Jan 16 2025",
                 "Spot Curve for Jan 17 2025"]
for element in l:
    df_spot[element] = []

df_spot["Name"] = namelist
df_spot['Maturity Date']=df['Maturity Date']
for t in range(0,10):
    spots=[]
    for T in range(0,10):
        #print(df.iloc[T, 2])
        #print (spots)
        spots.append(solve_for_r(spots,df.iloc[T, 2],df.iloc[T, 3],t,df_dirty.iloc[T, t+2 ]))
    df_spot[l[t+2]] = spots
    #print(df_spot[l[t+2]])
save = df_spot
#################################################################################################################
#notic that the spot curve is from
######we need to find one year rate
oney_spot = ['CAD spot rate for 1', '1/17/2030']
for i in range (0,10):
    oney_spot.append( tw_average(df_spot.iloc[1, i +2], df_spot.iloc[2, i +2], '9/1/2025','3/1/2026', "1/17/2026"))
a= df_spot.iloc[:2]
#print(a)
b =df_spot[2:]
#print (b)

c= {}
c = pd.DataFrame(c)
gamma = 0
for element in l:
    #print(oney_spot[gamma])
    c[element] = [oney_spot[gamma]]
    gamma += 1





#print(c)
d = pd.concat([a,c], axis = 0, ignore_index= True )
#print(d)
df_spot = pd.concat([d,b], axis = 0, ignore_index= True )
#print(df_spot)

########################################################
# notice that the spot curve is not perfect, we need to extend it to 5 year
df_lastspot = pd.DataFrame({})
df_lastspot['Maturity Date'] = ['1/17/2030']
df_lastspot["Name"] = ['5 year Spot']


for i in range(0,10):
    df_lastspot[l[i+2]] = [extract_line(df_spot.iloc[9, i+2], df_spot.iloc[10,i +2], '3/1/2029', '9/1/2029','1/17/2030')]
df_spot = pd.concat([df_spot,df_lastspot], axis = 0, ignore_index= True )



####################################################################################

#######################################################################################YTM plot
#print(df_YTM.columns) result = pd.concat([df1, df2], axis=0, ignore_index=True)
#print(day_count('9/1/2029')-365*2029-monthtoday(9))
#print(numdaysfromindex('9/1/2029', 9))
pay = payments_annu("9/1/2029", 9)
#note we need to add extentions as well
pay.append(5)
pay.insert(0,0)
#print(pay)
pl.title('Five Year Yield Curves')
for i in lt[2:]:
    y = df_YTM[i]
    pl.plot(pay,y,'o-', label= i )  # Plot the curve
pl.legend()
pl.grid(True)
pl.xlabel("Years (from Jan 17 2025)")
pl.ylabel("Yield to Maturity")
pl.show()
##################################################################################spot rate plot

lim = [1]
for i in range (2,10):
    lim.append(payments_annu('9/1/2029', 9)[i])
lim.append(5)
#pl.figure(1)

pl.title('Spot Rate Curves')
for i in range(0,10):
    y = df_spot.iloc[2:,i+2]
    pl.plot(lim,y,'o-', label = l[i+2])  # Plot the curve
pl.xlabel("Years")
pl.ylabel("Spot Rate")
pl.legend()
pl.grid(True)
pl.show()


############################################################################
#now we forward rate for this we will require spot rates for exact years
# we must choose half year point first July 18, 2025, where
#df_one year_estimates
df_oney_est = pd.DataFrame({})
col_one_est =["1,1","1,2", "1,3", "1,4"]

#for element in col_one_est:
#p#rint(df['Maturity Date'])


#build dataframe for one 1,2,3,4,5 spot estimates
#forward rate
#notic that the spot curve is from
######we need to find one year rate


mega_list=[]


for year in range(0,4):
    one_spot = []
    for i in range (0,10):
        one_spot.append( tw_average(save.iloc[2*year+1, i +2], save.iloc[2*year+2, i +2], save.iloc[2*year+1, 1],save.iloc[2*year +2, 1], ("1/17/202" + str( year +6))))
    mega_list.append(one_spot)

# for 5 th year we already solved before so we will append it
#print(df_spot)
mega_list.append(list(df_lastspot.iloc[0,2:]))

df_forward= pd.DataFrame({})
list_of_foreward = ["1-1","1-2",'1-3','1-4']
for element in ["1-1","1-2",'1-3','1-4']:
    df_forward[element] = []
###################################################calculation of forward rate.
# megalist[year][day]
for year in range(0,4):# for every year (from 1 to 5)
    templist = []
    for day in range(0,10):# iterate over 10 chosen days
        templist.append((1 + mega_list[year + 1][day]) ** (year+ 2) / (1 + mega_list[year][day]) ** (year+1) - 1) # computing 1 year forward rates
    df_forward[ "1-" + str( year+1)]  = templist
#print(df_forward)
pl.title('Forward Rate Curves')
for i in range(0,10):
    y = list(df_forward.iloc[i,0:])
    pl.plot(list_of_foreward, y,'o-', label= " forward rate curve for " + str(days[i]) +"   of Jan")  # Plot the curve
pl.legend()
pl.grid(True)
pl.xlabel("Years -Years")
pl.ylabel("Forward rate")
pl.show()
######################################################################################################################
# now we have have to produce values for one year yeilds:
# we will build a df with these interpolated values:
mega_list_sec=[]
print("Save:")
print(Save.columns)
for year in range(0,4):
    one_spot = []
    for i in range (0,10):
        one_spot.append( tw_average(Save.iloc[2*year+1, i +2], Save.iloc[2*year+2, i +2], Save.iloc[2*year+1, 1],Save.iloc[2*year +2, 1], ("1/17/202" + str( year +6))))

    mega_list_sec.append(one_spot)
mega_list.append(list(df_lastYTM.iloc[0,2:]))

df_oneyield= pd.DataFrame({})
list_oneyield = ["Year 1 yeild"," Year 2 yeild",'Year 3 yeild','4 yeild']
for element in list_oneyield:
    df_oneyield[element] = []
# notice that we need to add the fith year which i will add now:
mega_list_sec.append(list(df_lastYTM.iloc[0,2:]))


#we will now build an array of X[i][j] the j = 1,2,3...9,  i = 1,2,3,4,5
Xij_Y = pd.DataFrame({})

for i in range(1,6):
   Xij_Y[str(i)] = []
#print(Xij_Y)

for i in range(1,6):
    xii = []

    for j in range(1,10):# 1 to 5

        xii.append( math.log(mega_list_sec[i-1][j+1-1]/(mega_list_sec[i-1][j-1])))# there are some problems
    Xij_Y[str(i)] = xii
# here is the covarience matrix
print(Xij_Y.cov())
###########################################################################################################################
#forward rate df:################################




Xij_f = pd.DataFrame({})

for i in range(1,5):
   Xij_f["1-"+ str(i)] = []
#print(Xij_Y)

for i in range(1,5):
    xii = []

    for j in range(1,10):# 1 to 5

        xii.append( math.log(df_forward.iloc[j-1+1, i-1]/(df_forward.iloc[j-1,i-1])))# there are some problems
    Xij_f["1-"+str(i)] = xii
# here is the covarience matrix
print(Xij_f.cov())
###################################################################
####eigen vectors for Xij_y
matrix = (Xij_Y.cov()).to_numpy()

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(matrix)

# Display results
print("Eigenvalues for Xij_Y:")
print(eigenvalues)

print("\nEigenvectors: Xij_Y")
print(eigenvectors)
##############################################################
#######eigen vectors for Xij_f
matrix = (Xij_f.cov()).to_numpy()

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(matrix)

# Display results
print("Eigenvalues for forward.cov:")
print(eigenvalues)

print("\nEigenvectors:forward.cov")
print(eigenvectors)

