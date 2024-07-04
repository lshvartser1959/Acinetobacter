import numpy as np
import pandas as pd

def rmv_last(str1, fi):
    lst = fi["Name"].values
    res = [str for str in lst if str1 in str]
    res1 = [str[:-len(str1)-1] for str in lst if str1 in str]
    fi["Name"] = fi["Name"].replace(res, res1)

    return fi

gap_times = 6
fi = pd.read_csv('feature_impo_6.csv')

fi = fi.set_axis(["Name", "Val"], axis=1, inplace = False)

fi = rmv_last('mean_1', fi)
fi = rmv_last('mask_1', fi)
fi = rmv_last('time_since_measured_1', fi)

fi = rmv_last('mean_2', fi)
fi = rmv_last('mask_2', fi)
fi = rmv_last('time_since_measured_2', fi)

fi = rmv_last('mean_3', fi)
fi = rmv_last('mask_3', fi)
fi = rmv_last('time_since_measured_3', fi)

fi = rmv_last('mean_4', fi)
fi = rmv_last('mask_4', fi)
fi = rmv_last('time_since_measured_4', fi)

fi = rmv_last('mean_5', fi)
fi = rmv_last('mask_5', fi)
fi = rmv_last('time_since_measured_5', fi)

fi = rmv_last('mean', fi)
fi = rmv_last('mask', fi)
fi = rmv_last('time_since_measured', fi)

fi_sum = fi.groupby('Name').sum().sort_values(by=['Val'],ascending=False)

fi_sum.to_csv('/mnt/data2/'+'sum_feature_impo_'+ str(gap_times)+'.csv')
aa=0