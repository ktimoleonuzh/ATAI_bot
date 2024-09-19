# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 18:06:12 2022

@author: Nadia Timoleon
"""
from agent import MyBot
from load_all_data import graph, image_data


# username = 'lazyLlama7_bot'
# password = 'K2FchcpW2--0xg'
username = 'konstantina.timoleon_bot'
password = 'K2FchcpW2--0xg'
mybot = MyBot(username, password, graph, image_data)
#mybot.listen()