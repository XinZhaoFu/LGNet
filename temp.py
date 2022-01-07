from datetime import datetime


start_time = datetime.now()

start_time = str(start_time)[:19]
tran_tab = str.maketrans('- :', '___')
plt_name = str(start_time).translate(tran_tab)

print(plt_name, start_time)
