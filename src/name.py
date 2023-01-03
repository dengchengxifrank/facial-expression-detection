import os
import json

data_path = './dataset'
data_dir_list = os.listdir(data_path)
lable_list={}
cnt = 0

angry = [1,0,0,0,0,0,0]
disgust = [0,1,0,0,0,0,0]
fear = [0,0,1,0,0,0,0]
happy = [0,0,0,1,0,0,0]
sad = [0,0,0,0,1,0,0]
surprise = [0,0,0,0,0,1,0]
neutral = [0,0,0,0,0,0,1]
cnt = 1

for i in data_dir_list:
    print(('D:\\Desktop\\code\\dataset'+'\\'+i))
    if i[3]+i[4] == 'AN':
        os.rename('D:\\Desktop\\code\\dataset'+'\\'+i,'D:\\Desktop\\code\\dataset'+'\\'+str(cnt)+'.tiff')
        lable_list[cnt] = angry
        cnt = cnt + 1

    elif i[3]+i[4] == 'DI':
        os.rename('D:\\Desktop\\code\\dataset'+'\\'+i,'D:\\Desktop\\code\\dataset'+'\\'+str(cnt)+'.tiff')
        lable_list[cnt] = disgust
        cnt = cnt + 1


    elif i[3]+i[4] == 'FE':

        os.rename('D:\\Desktop\\code\\dataset'+'\\'+i,'D:\\Desktop\\code\\dataset'+'\\'+str(cnt)+'.tiff')
        lable_list[cnt] = fear
        cnt = cnt + 1



    elif i[3]+i[4] == 'HA':
        os.rename('D:\\Desktop\\code\\dataset'+'\\'+i,'D:\\Desktop\\code\\dataset'+'\\'+str(cnt)+'.tiff')

        lable_list[cnt] = happy
        cnt = cnt + 1


    elif i[3]+i[4] == 'NE':
        os.rename('D:\\Desktop\\code\\dataset'+'\\'+i,'D:\\Desktop\\code\\dataset'+'\\'+str(cnt)+'.tiff')

        lable_list[cnt] = neutral
        cnt = cnt + 1


    elif i[3]+i[4] == 'SA':

        os.rename('D:\\Desktop\\code\\dataset'+'\\'+i,'D:\\Desktop\\code\\dataset'+'\\'+str(cnt)+'.tiff')

        lable_list[cnt] = sad

        cnt = cnt + 1

    elif i[3]+i[4] == 'SU':
        os.rename('D:\\Desktop\\code\\dataset'+'\\'+i,'D:\\Desktop\\code\\dataset'+'\\'+str(cnt)+'.tiff')

        lable_list[cnt] = surprise

        cnt = cnt + 1


tf = open("./save.json",'r')
new = json.load(tf)
print(new)

