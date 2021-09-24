# streamlit run app.py
# pip install -r requirements.txt

import streamlit as st
import pickle
import pandas as pd
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# st.title('Fraud Detection')
# st.markdown("<h1 style='text-align: center; color: black;'>Fraud Detection</h1>", unsafe_allow_html=True)
im = Image.open("cover.png")
st.image(im, width=700)

html_temp = """
<div style="width:700px;background-color:maroon;padding:10px">
<h1 style="color:white;text-align:center;">Machine Learning Application (Demo)</h1>
</div>"""
st.markdown(html_temp,unsafe_allow_html=True)

# # Setting Application title
# st.title('Machine Learning Application (Demo)')

# Setting Application description
st.markdown("""
 :dart:  This Streamlit app is made to predict credit card fraud detection in a finance company use case. 
 The application is functional for both online prediction and batch data prediction.. \n
""")
st.markdown("<h3></h3>", unsafe_allow_html=True)

# # images
# im = Image.open("image.png")
# st.image(im, width=700)

def main():
    st.sidebar.header("How would you like to predict?")
    add_selectbox = st.sidebar.selectbox("", ("Unique Input", "Batch Input"))
    st.sidebar.info('This app is created to predict credit card fraud detection.')

    features= ['card1', 'card2', 'card4', 'card5', 
               'D1', 'D3', 'D4', 'D5', 'D8', 'D10', 'D11', 'D15', 
               'C1', 'C5', 'C13', 
               'R_emaildomain_1', 'P_emaildomain_1', 
               'TransactionAmt', 'addr1', 'dist1', 
               'Transaction_hour', 'Transaction_day_of_week', 
               'id_19', 'id_20', 'id_33']

    id_33 = [2073600,  230400,   270000,   384480,   410400,   442368,
             480000,   518400,   565248,   585600,   614400,   636411,
             655360,   669060,   670488,   672195,   727040,   728320,
             746496,   768000,   784000,   784641,   785408,   785664,
             786432,   787200,   793600,   810000,   829440,   838656,
             849239,   852480,   855360,   863583,   864000,   911360,
             921600,   922320,   923601,   936960,   947200,   983040,
             985600,   995328,  1000500,  1003104,  1016064,  1024000,
            1024800,  1026081,  1036800,  1043120,  1043712,  1044480,
            1046955,  1047552,  1047722,  1048320,  1049088,  1050624,
            1065600,  1092960,  1115136,  1128960,  1133160,  1152000,
            1156320,  1164151,  1166400,  1167936,  1174947,  1220400,
            1221120,  1228800,  1239040,  1260000,  1262624,  1264158,
            1267688,  1269190,  1293661,  1294560,  1295100,  1296000,
            1298341,  1308417,  1309440,  1309696,  1310720,  1312000,
            1327104,  1339200,  1382400,  1398784,  1426384,  1428840,
            1437501,  1438400,  1439100,  1440000,  1470000,  1474560,
            1510236,  1568160,  1568468,  1587600,  1594748,  1600000,
            1632000,  1638400,  1638720,  1676917,  1679616,  1682317,
            1726316,  1742400,  1753856,  1753920,  1761271,  1762320,
            1764000,  1765680,  1769472,  1776000,  1776608,  1832075,
            1871424,  1872000,  1918080,  1920000,  1954560,  1959800,
            2025000,  2070601,  2071440,  2071680,  2072520,  153600,
            2074680,  2075520,  2076601,  2134440,  2170800,  2211840,
            2242080,  2250000,  2251125,  2300881,  2302800,  2304000,
            2305920,  2332800,  2334960,  2359296,  2360448,  2397600,
            2399820,  2455680,  2457600,  2459520,  2476800,  2509056,
            2617344,  2621440,  2624400,  2740500,  2742336,  2755200,
            2764800,  2916000,  2985984,  3106801,  3108240,  3108960,
            3110400,  3145728,  3147264,  3240000,  3317760,  3389265,
            3393024,  3409920,  3412024,  3444480,  3528000,  3600000,
            3682401,  3684960,  3686400,  3692962,  3694404,  3709632,
            3840000,  3847680,  3902400,  3969000,  4096000,  4147200,
            4152960,  4155844,  4194829,  4262400,  4263840,  4269762,
            4352000,  4410000,  4460544,  4608000,  4665600,  4860000,
            4953600,  4955040,  4956160,  4985905,  4987728,  4990464,
            4991375,  4995025,  5184000,  5308416,  5595136,  5758200,
            5760000,  5761800,  5998000,  6000000,  6002000,  6144000,
            6220800,  6350400,  6400000,  6969600,  6998400,  7056000,
            8100000,  8282404,  8288401,  8290080,  8292240,  8294400,
            8300401,  8302080,  8312409,  8537760,  9216000,  9437184,
           11025000, 13492501, 13500000, 14745600, 15876000, 18662400,
           20358144, 23040000, 25401600, 37500000]


    R_emaildomain_1_list = ['None', 'gmail', 'hotmail', 'outlook', 'anonymous', 'charter',
                           'prodigy', 'comcast', 'live', 'icloud', 'yahoo', 'aol', 'juno',
                           'att', 'verizon', 'bellsouth', 'servicios-ta', 'ymail', 'msn',
                           'optonline', 'gmx', 'aim', 'mail', 'me', 'cox', 'earthlink',
                           'embarqmail', 'web', 'sbcglobal', 'scranton', 'mac', 'twc',
                           'roadrunner', 'frontiernet', 'q', 'windstream', 'suddenlink',
                           'ptd', 'frontier', 'cfl', 'netzero', 'rocketmail', 'centurylink',
                           'protonmail', 'cableone', 'sc']
    P_emaildomain_1_list = ['gmail', 'outlook', 'yahoo', 'mail', 'anonymous',
                           'hotmail', 'verizon', 'aol', 'me', 'comcast', 'optonline', 'cox',
                           'charter', 'rocketmail', 'prodigy', 'embarqmail', 'icloud', 'live',
                           'att', 'juno', 'ymail', 'sbcglobal', 'bellsouth', 'msn', 'q',
                           'centurylink', 'servicios-ta', 'earthlink', 'cfl', 'roadrunner',
                           'netzero', 'gmx', 'suddenlink', 'frontiernet', 'windstream',
                           'frontier', 'mac', 'aim', 'web', 'twc', 'cableone', 'sc', 'ptd',
                           'protonmail','None']



    all_columns = ['TransactionAmt', 'card1', 'card2', 'card3', 'card5', 'addr1', 'dist1',
           'C1', 'C5', 'C13', 'D1', 'D3', 'D4', 'D5', 'D8', 'D9', 'D10', 'D11',
           'D13', 'D14', 'D15', 'id_01', 'id_02', 'id_03', 'id_05', 'id_06',
           'id_09', 'id_10', 'id_11', 'id_13', 'id_14', 'id_17', 'id_19', 'id_20',
           'id_32', 'id_33', 'Transaction_hour', 'Transaction_day_of_week',
           'TransactionAmt_to_mean_card1', 'TransactionAmt_to_mean_card4',
           'TransactionAmt_to_std_card1', 'TransactionAmt_to_std_card4',
           'id_02_to_mean_card1', 'id_02_to_mean_card4', 'id_02_to_std_card1',
           'id_02_to_std_card4', 'D15_to_mean_card1', 'D15_to_mean_card4',
           'D15_to_std_card1', 'D15_to_std_card4', 'D15_to_mean_addr1',
           'D15_to_std_addr1', 'TransactionAmt_decimal',
           'TransactionAmt_winsorize', 'TransactionAmt_log', 'TransactionAmt_sqrt',
           'TransactionAmt_z', 'pca_V1', 'pca_V2', 'ProductCD_freq',
           'DeviceInfo_freq', 'DeviceType_freq', 'R_emaildomain_1_freq',
           'R_emaildomain_2_freq', 'R_emaildomain_3_freq', 'P_emaildomain_1_freq',
           'P_emaildomain_2_freq', 'P_emaildomain_3_freq', 'id_12_freq',
           'id_28_freq', 'id_30_freq', 'id_31_freq', 'id_34_freq', 'id_35_freq',
           'id_36_freq', 'id_37_freq', 'id_38_freq', 'M2_freq', 'M3_freq',
           'M4_freq', 'M5_freq', 'M6_freq', 'M7_freq', 'M8_freq', 'M9_freq',
           'card4_freq', 'card6_freq']

    # @st.cache
    # bir buyuk bir datatyi read_csv ile tekrar tekrar okutmamak icin hafuzada tutmasi icin st.cache kullanilir.
    lightGBM = pickle.load(open("lightGBM.pkl","rb"))
    
    with open('FE_dict.pkl', 'rb') as handle:
        FE_dict = pickle.load(handle)
    
    ############################################################################################################
    if add_selectbox == "Unique Input":
        card1 = st.slider("Card1:", 1000,18397,8732, step=1)
        card2 = st.slider("Card2:", 100,601,360, step=1)
        card4 = st.selectbox("Card4:",('mastercard','american express','discover','visa','None'))
        card5 = st.slider("Card5:", 100,238,229, step=1)

        id_19 = st.slider("id_19:", 100,672,341, step=1)
        id_20 = st.slider("id_20:", 100,662,472, step=1)
        id_33 = st.selectbox("id_33",(id_33)) #kategorik yap

        D1 = st.slider("D1:", 0,641,0, step=1)
        D3 = st.slider("D3:", 0,820,1, step=1)
        D4 = st.slider("D4:", -122,670,1, step=1)
        D5 = st.slider("D5:", 0,820,0, step=1)
        D8 = st.slider("D8:", 0,1708,37, step=1)
        D10 = st.slider("D10:", 0,877,0, step=1)
        D11 = st.slider("D11:", -53,671,0, step=1)
        D15 = st.slider("D15:", -83,880,0, step=1)

        C1 = st.slider("C1:", 0,4686,2, step=1)
        C5 = st.slider("C5:", 0,350,0, step=1)
        C13 = st.slider("C13:", 0,2919,0, step=1)

        TransactionAmt = st.slider("TransactionAmt:", 0,31938,126, step=1)
        addr1 = st.slider("addr1:", 0,541,330, step=1)
        dist1 = st.slider("dist1:", 0,10287,8, step=1)
        Transaction_hour = st.slider("Transaction_hour:", 0,23,22, step=1)
        Transaction_day_of_week = st.slider("Transaction_day_of_week:", 1,7,1, step=1)-1


        R_emaildomain_1 = st.selectbox("R_emaildomain_1:",(R_emaildomain_1_list))
        P_emaildomain_1 = st.selectbox("P_emaildomain_1:",(P_emaildomain_1_list))

        my_dict = {'C1': C1,
                     'C13': C13,
                     'C5': C5,
                     'D1': D1,
                     'D10': D10,
                     'D11': D11,
                     'D15': D15,
                     'D3': D3,
                     'D4': D4,
                     'D5': D5,
                     'D8': D8,
                     'P_emaildomain_1': P_emaildomain_1,
                     'R_emaildomain_1': R_emaildomain_1,
                     'TransactionAmt': TransactionAmt,
                     'Transaction_day_of_week': Transaction_day_of_week,
                     'Transaction_hour': Transaction_hour,
                     'addr1': addr1,
                     'card1': card1,
                     'card2': card2,
                     'card4': card4,
                     'card5': card5,
                     'dist1': dist1,
                     'id_19': id_19,
                     'id_20': id_20,
                     'id_33': id_33}

        with open('FE_dict.pkl', 'rb') as handle:
            FE_dict = pickle.load(handle)

        # Feature Engineering
        my_dict['TransactionAmt_to_std_card1'] = FE_dict['TransactionAmt_to_std_card1'][my_dict['card1']]
        my_dict['TransactionAmt_to_mean_card1'] = FE_dict['TransactionAmt_to_mean_card1'][my_dict['card1']]
        my_dict['TransactionAmt_to_std_card4'] = FE_dict['TransactionAmt_to_std_card4'][my_dict['card4']]
        my_dict['D15_to_mean_card1'] = FE_dict['D15_to_mean_card1'][my_dict['card1']]
        my_dict['D15_to_std_card1'] = FE_dict['D15_to_std_card1'][my_dict['card1']]
        my_dict['TransactionAmt_to_mean_card4'] = FE_dict['TransactionAmt_to_mean_card4'][my_dict['card4']]
        my_dict['D15_to_mean_addr1'] = FE_dict['D15_to_mean_addr1'][my_dict['addr1']]

        my_dict['TransactionAmt_decimal'] = (my_dict['TransactionAmt']-np.floor(my_dict['TransactionAmt']))*100

        # Frequency Encoding
        my_dict['P_emaildomain_1_freq'] = FE_dict['P_emaildomain_1_freq'][my_dict['P_emaildomain_1']]
        my_dict['R_emaildomain_1_freq'] = FE_dict['R_emaildomain_1_freq'] [my_dict['R_emaildomain_1']]
        my_dict['card4_freq'] = FE_dict['card4_freq'][my_dict['card4']]                                                                      

        df = pd.DataFrame([my_dict]).drop(['card4', 'R_emaildomain_1', 'P_emaildomain_1'], axis=1)

        all_columns = ['TransactionAmt', 'card1', 'card2', 'card3', 'card5', 'addr1', 'dist1',
               'C1', 'C5', 'C13', 'D1', 'D3', 'D4', 'D5', 'D8', 'D9', 'D10', 'D11',
               'D13', 'D14', 'D15', 'id_01', 'id_02', 'id_03', 'id_05', 'id_06',
               'id_09', 'id_10', 'id_11', 'id_13', 'id_14', 'id_17', 'id_19', 'id_20',
               'id_32', 'id_33', 'Transaction_hour', 'Transaction_day_of_week',
               'TransactionAmt_to_mean_card1', 'TransactionAmt_to_mean_card4',
               'TransactionAmt_to_std_card1', 'TransactionAmt_to_std_card4',
               'id_02_to_mean_card1', 'id_02_to_mean_card4', 'id_02_to_std_card1',
               'id_02_to_std_card4', 'D15_to_mean_card1', 'D15_to_mean_card4',
               'D15_to_std_card1', 'D15_to_std_card4', 'D15_to_mean_addr1',
               'D15_to_std_addr1', 'TransactionAmt_decimal',
               'TransactionAmt_winsorize', 'TransactionAmt_log', 'TransactionAmt_sqrt',
               'TransactionAmt_z', 'pca_V1', 'pca_V2', 'ProductCD_freq',
               'DeviceInfo_freq', 'DeviceType_freq', 'R_emaildomain_1_freq',
               'R_emaildomain_2_freq', 'R_emaildomain_3_freq', 'P_emaildomain_1_freq',
               'P_emaildomain_2_freq', 'P_emaildomain_3_freq', 'id_12_freq',
               'id_28_freq', 'id_30_freq', 'id_31_freq', 'id_34_freq', 'id_35_freq',
               'id_36_freq', 'id_37_freq', 'id_38_freq', 'M2_freq', 'M3_freq',
               'M4_freq', 'M5_freq', 'M6_freq', 'M7_freq', 'M8_freq', 'M9_freq',
               'card4_freq', 'card6_freq']

        df = df.reindex(columns=all_columns, fill_value=0)

        # Table
        def single_customer(my_dict):
            df_table = pd.DataFrame.from_dict([my_dict])
        #     st.table(df_table) 
            st.write('')
            st.dataframe(data=df_table, width=700, height=400)
            st.write('')

        single_customer(my_dict)

        # Button
        if st.button("Submit Manuel Inputs"):
            import time
            with st.spinner("ML Model is loading..."):
                my_bar=st.progress(0)
                for p in range(0,101,10):
                    my_bar.progress(p)
                    time.sleep(0.1)

                    fraud_probability = lightGBM.predict_proba(df)
                    is_fraud= lightGBM.predict(df)

                st.success(f'The Fraud Probability of the Transaction is {round(fraud_probability[0][1]*100,3)}%')
        #         st.success(f'The Fraud Probability of the Transaction is {fraud_probability[0][1]}')

                if is_fraud[0]:
                    st.success("The Transaction is FRAUD")
                else:
                    st.warning("The Transaction is NOT FRAUD")
    
    else:
        # Upload a csv
        output = pd.DataFrame()
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            file = pd.read_csv(uploaded_file)
            flag=file.copy()
            st.dataframe(data=file, width=700, height=1000)
            st.write('')
        #  st.table(file)
        
        # Load Button
        if st.button("Submit CSV File"):
            import time
            with st.spinner("ML Model is loading..."):
                my_bar=st.progress(0)
                for p in range(0,101,10):
                    my_bar.progress(p)
                    time.sleep(0.1)

            for i in file.index:
                # Feature Engineering
                file.loc[i,'TransactionAmt_to_std_card1'] = FE_dict['TransactionAmt_to_std_card1'][file.loc[i,'card1']]
                file.loc[i,'TransactionAmt_to_mean_card1'] = FE_dict['TransactionAmt_to_mean_card1'][file.loc[i,'card1']]
                file.loc[i,'TransactionAmt_to_std_card4'] = FE_dict['TransactionAmt_to_std_card4'][file.loc[i,'card4']]
                file.loc[i,'D15_to_mean_card1'] = FE_dict['D15_to_mean_card1'][file.loc[i,'card1']]
                file.loc[i,'D15_to_std_card1'] = FE_dict['D15_to_std_card1'][file.loc[i,'card1']]
                file.loc[i,'TransactionAmt_to_mean_card4'] = FE_dict['TransactionAmt_to_mean_card4'][file.loc[i,'card4']]
                file.loc[i,'D15_to_mean_addr1'] = FE_dict['D15_to_mean_addr1'][file.loc[i,'addr1']]

                file.loc[i,'TransactionAmt_decimal'] = (file.loc[i,'TransactionAmt']-np.floor(file.loc[i,'TransactionAmt']))*100

                # Frequency Encoding
                file.loc[i,'P_emaildomain_1_freq'] = FE_dict['P_emaildomain_1_freq'][file.loc[i,'P_emaildomain_1']]
                file.loc[i,'R_emaildomain_1_freq'] = FE_dict['R_emaildomain_1_freq'] [file.loc[i,'R_emaildomain_1']]
                file.loc[i,'card4_freq'] = FE_dict['card4_freq'][file.loc[i,'card4']]                                                           

            file = file.drop(['card4', 'R_emaildomain_1', 'P_emaildomain_1'], axis=1)
            file = file.reindex(columns=all_columns, fill_value=0)
            pred_file= pd.DataFrame(lightGBM.predict_proba(file))[[1]].rename({1:'Prediction'}, axis=1)
            pred_file['isFraud'] = pred_file.iloc[:,0].apply(lambda x: 'YES' if np.float(x)>=0.5 else 'NO')
            output = pd.concat([pred_file,flag], axis=1)
            st.write('')
            st.dataframe(data=output, width=700, height=400)
            st.write('')

        def download_link(object_to_download, download_filename, download_link_text):
            if isinstance(object_to_download,pd.DataFrame):
                object_to_download = object_to_download.to_csv(index=False)

            # some strings <-> bytes conversions necessary here
            b64 = base64.b64encode(object_to_download.encode()).decode()

            return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

        # output = pd.concat([pred_file,flag], axis=1)

        # if st.button('Download Output as CSV'):
        tmp_download_link = download_link(output, 'output.csv', 'Click here to download output as csv!')
        st.markdown(tmp_download_link, unsafe_allow_html=True)

        comment = st.text_input('Write your comments below.')
        # st.write(comment)

        # if st.button('Download input as a text file'):
        tmp_download_link = download_link(comment, 'commend.txt', 'Click here to download comment text!')
        st.markdown(tmp_download_link, unsafe_allow_html=True)

if __name__ == '__main__':
    main()