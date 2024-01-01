import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import association_rules, apriori

bakery = pd.read_csv('BreadBasket_DMS.csv')
bakery['Datetime'] = pd.to_datetime(bakery['Date'] , format='%Y-%m-%d')


bakery["month"] = bakery['Datetime'].dt.month
bakery["day"] = bakery['Datetime'].dt.day

bakery["month"].replace([i for i in range(1, 12 + 1)], ["Januari","Februari","Maret","April","Mei","Juni","Juli","Agustur","September","Oktober","November","Desember"], inplace=True)
bakery["day"].replace([i for i in range(1,8)], ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"],inplace=True)

st.title("Transaction from a bakery")

def get_data( month ='' , day = ''):
    data = bakery.copy()
    filtered = data.loc[
        (data["month"].str.contains(month.title())) &
        (data["day"].str.contains(day.title()))
    ]
    return filtered if filtered.shape[0] else "No Result!"

def user_input_features():
    item = st.selectbox("Item", ['Bread', 'Scandinavian', 'Hot chocolate', 'Jam', 'Cookies',
       'Muffin', 'Coffee', 'Pastry', 'Medialuna', 'Tea', 'NONE',
       'Tartine', 'Basket', 'Mineral water', 'Farm House', 'Fudge',
       'Juice', "Ella's Kitchen Pouches", 'Victorian Sponge', 'Frittata',
       'Hearty & Seasonal', 'Soup', 'Pick and Mix Bowls', 'Smoothies',
       'Cake', 'Mighty Protein', 'Chicken sand', 'Coke',
       'My-5 Fruit Shoot', 'Focaccia', 'Sandwich', 'Alfajores', 'Eggs',
       'Brownie', 'Dulce de Leche', 'Honey', 'The BART', 'Granola',
       'Fairy Doors', 'Empanadas', 'Keeping It Local', 'Art Tray',
       'Bowl Nic Pitt', 'Bread Pudding', 'Adjustment', 'Truffles',
       'Chimichurri Oil', 'Bacon', 'Spread', 'Kids biscuit', 'Siblings',
       'Caramel bites', 'Jammie Dodgers', 'Tiffin', 'Olum & polenta',
       'Polenta', 'The Nomad', 'Hack the stack', 'Bakewell',
       'Lemon and coconut', 'Toast', 'Scone', 'Crepes', 'Vegan mincepie',
       'Bare Popcorn', 'Muesli', 'Crisps', 'Pintxos', 'Gingerbread syrup',
       'Panatone', 'Brioche and salami', 'Afternoon with the baker',
       'Salad', 'Chicken Stew', 'Spanish Brunch',
       'Raspberry shortbread sandwich', 'Extra Salami or Feta',
       'Duck egg', 'Baguette', "Valentine's card", 'Tshirt',
       'Vegan Feast', 'Postcard', 'Nomad bag', 'Chocolates',
       'Coffee granules ', 'Drinking chocolate spoons ',
       'Christmas common', 'Argentina Night', 'Half slice Monster ',
       'Gift voucher', 'Cherry me Dried fruit', 'Mortimer', 'Raw bars',
       'Tacos/Fajita'])
    month = st.select_slider("Month", ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"])
    day = st.select_slider("Day", ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"])

    
    return item, month, day

item, month, day = user_input_features()

data = get_data(month, day)

def encode(x):
    if x <= 0:
        return 0
    elif x >= 1:
        return 1
    
if type(data) != type ("No Result"):
    item_count = data.groupby(['Transaction', 'Item'])["Item"].count().reset_index(name="Count")
    item_count_pivot = item_count.pivot_table(index='Transaction', columns='Item', values='Count', aggfunc='sum').fillna(0) 
    item_count_pivot = item_count_pivot.applymap(encode)

    support = 0.05
    frequent_items = apriori(item_count_pivot, min_support=support, use_colnames=True)

    metric = "lift"
    min_threshold = 1

    rules = association_rules(frequent_items, metric=metric, min_threshold=min_threshold)[["antecedents","consequents","support","confidence","lift"]]
    rules.sort_values('confidence', ascending=False,inplace=True)

def parse_list(x):
    x=list(x)
    if len(x) == 1:
        return x[0]
    elif len(x) > 1:
        return ", ".join(x)

def return_item_df(item_antecedents):
    
    data=rules[["antecedents", "consequents"]].copy()
    
    data["antecedents"] = data["antecedents"].apply(parse_list)
    data["consequents"] = data["consequents"].apply(parse_list)
    
    filtered_data = data.loc[data["antecedents"] == item_antecedents]

    if not filtered_data.empty:
        return list(filtered_data.iloc[0, :])
    else:
        return []

if type(data) != type("No Result!"):
    st.markdown("Hasil Rekomendasi : ")
    result = return_item_df(item)
    if result :
        st.success(f"Jika Konsumen Membeli **{item}**, maka membeli **{return_item_df(item)[1]}** secara bersamaan")
    else:
        st.warning("Tidak ditemukan rekomendasi untuk item yang dipilih")

