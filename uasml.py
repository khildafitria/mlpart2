import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import association_rules, apriori

bakery = pd.read_csv('BreadBasket_DMS.csv')
bakery['Datetime'] = pd.to_datetime(bakery['Date'] + ' ' + bakery['Time'], format='%Y-%m-%d %H:%M:%S')
bakery['Datetime'] = pd.to_datetime(bakery['Datetime'], format= "%d-%m-%Y")

bakery["month"] = bakery['Datetime'].dt.month
bakery["day"] = bakery['Datetime'].dt.day

bakery["month"].replace([i for i in range(1, 12 + 1)], ["Januari","Februari","Maret","April","Mei","Juni","Juli","Agustur","September","Oktober","November","Desember"], inplace=True)
bakery["day"].replace([i for i in range(6 + 1)], ["senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"],inplace=True)

st.title("UAS Transaction from a bakery Algoritma Apriori")

def get_bakery( month ='' , day = ''):
    bakery = bakery.copy()
    filtered = bakery.loc[
        (bakery["month"].str.contains(month.title())) &
        (bakery["day"].str.contains(day.title()))
    ]
    return filtered if filtered.shape[0] else "No Result!"

def user_input_features():
    item = st.selectbox("Item", ['coffee', 'bread', 'tea', 'cake', 'pastry', 'none', 'sandwich', 'medialuna', 'hot chocolate', 'cookies', 'brownie', 'farm house', 'muffin', 'juice', 'alfajores', 'soup', 'scone', 'toast', 'scandinavian', 'truffles', 'coke', 'spanish brunch', 'fudge', 'baguette', 'jam', 'tiffin', 'mineral water', 'jammie' 'dodgers', 'chicken stew', 'hearty & seasonal', 'salad', 'frittata', 'smoothies', 'keeping it local', 'the nomad', 'focaccia', 'vegan mincepie', 'bakewell', 'tartine', 'afternoon with the baker', 'extra salami or feta', 'art tray', 'eggs', 'granola', 'tshirt', 'my-5 fruit shoot', 'ellas kitchen pouches', 'vegan feast', 'crisps', 'dulce de leche', 'valentines card', 'kids biscuit', 'duck egg', 'pick and mix bowls', 'christmas common', 'tacos/fajita', 'mighty protein', 'chocolates', 'postcard', 'gingerbread syrup', 'muesli nomad bag', 'drinking chocolate spoons', 'coffee granules', 'victorian sponge', 'empanadas', 'argentina night', 'crepes', 'honey', 'pintxos', 'lemon and coconut', 'basket', 'half slice monster', 'bare popcorn', 'panatone', 'mortimer', 'bread pudding', 'cherry me dried fruit', 'brioche and salami', 'caramel bites', 'raspberry short bread sandwich', 'fairy doors', 'hack the stack', 'bowl nic pitt', 'chimichurri oil', 'spread', 'siblings', 'gift voucher', 'raw bars', 'polenta', 'chicken sand', 'the bart', 'adjustment', 'olum & polenta', 'bacon'])
    month = st.select_slider("Month", ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"])
    day = st.select_slider("Day", ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"], value='Senin')

    return item, month, day

item, month, day = user_input_features()

bakery = get_bakery(month, day)

def encode(x):
    if x <= 0:
        return 0
    elif x >= 1:
        return 1
    
if type(bakery) != type ("No Result"):
    item_count = bakery.groupby(['Transaction', 'Item'])["Item"].count().reset_index(name="Count")
    item_count_pivot = item_count.pivot_table(index='Transaction', columns='Item', values='Count', aggfunc='sum').fillna(0) 
    item_count_pivot = item_count_pivot.applymap(encode)

    support = 0.01
    frequent_items = apriori(item_count_pivot, min_support=support, use_colnames=True)

    metric = "lift"
    min_threshold = 1

    rules = association_rules(frequent_items, metric=metric, min_threshold=min_threshold)[["antecedents","consequents","support","confidence","lift"]]
    rules.sort_values('confidence', ascending=False,inplace=True)

def parse_list(x):
    x = list(x)
    if len(x) == 1:
        return x[0]
    elif len(x) > 1:
        return ", ".join(x)

def return_item_bakery(item_antecedents):
    bakery = rules[["antecedents", "consequents"]].copy()
     
    bakery["antecedents"] = bakery["antecedents"].apply(parse_list)
    bakery["consequents"] = bakery["consequents"].apply(parse_list)

    return list(bakery.loc[bakery["antecedents"] == item_antecedents].iloc[0,:])

if type(bakery) != type("No Result!"):
    st.markdown("Hasil Rekomendasi : ")
    st.success(f"Jika Konsumen Membeli **{item}**, maka membeli **{return_item_bakery(item)[1]}** secara bersamaan")
    
