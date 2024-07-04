import numpy as np
import pandas as pd

if __name__ == '__main__':
    #read data
    Data=pd.read_excel('feature_impoLR_6_int.xlsx')

    names = Data['names'].copy()
    for i in range(len(names)):
        nam = names[i]
        send = nam.find(' mask')
        if send<0:
            send = nam.find(' mean')
            if send<0:
                send = nam.find(' time_since_measured')
        if send >= 0:
            names[i] = nam[0:send]

    names = np.unique(np.array(names))
    fi_bd = Data

    newDF = pd.DataFrame(columns=['features', 'weights'])

    for nam in names:
        # if nam == 'name solid tumor' or nam == 'microbiologic documentation':
        #     continue
        ind = np.where(fi_bd['names'].str.contains(nam))[0]
        if len(ind):
            weight = sum(fi_bd.loc[ind]['conf'])
            newDF = newDF.append({'features': nam, 'weights': weight}, ignore_index=True)

    newDF.to_excel('feature_importances_' + 'Beil_6_LR' + '.xlsx')

    ss=0

