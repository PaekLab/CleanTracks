import numpy as np
from hmmlearn import hmm
import pandas as pd

def HMM_correction(data_frame, trans_mat, em_mat, n_components = 2):
    start_probabilities = np.array([0.999, 0.001])  # Start in "live" state

    model = hmm.CategoricalHMM(n_components=n_components, init_params="")
    model.startprob_ = start_probabilities
    model.transmat_ = trans_mat
    model.emissionprob_ = em_mat

    final_arr = []
    score_arr = []
    for i in range(len(data_frame)):


        classifier_predictions = np.array([int(x) for x in data_frame[i]]).reshape(-1, 1)

    # Infer most likely hidden state sequence
        hidden_states = model.decode(classifier_predictions)
        #print(hidden_states)
    # Apply post-processing: all timepoints after first "dead" are set to dead
        #if 1 in hidden_states:
        #    first_death = np.argmax(hidden_states == 1)  # Find first dead state
        #    hidden_states[first_death:] = 1  # Set all timepoints after to dead
        final_arr.append(hidden_states[1])
        score_arr.append(hidden_states[0])


    #optionally, return a confidence score for each prediction
    #confidence_scores = model.predict_proba(classifier_predictions)

    final_arr = np.array(final_arr)
    return final_arr, score_arr

def trans_mat_from_data(dataframe, label_col, id_col, segmentation = "", label_val = 1):
    """trans_mat_from_data generates a transition matrix for the hidden states of a HMM from training data that is pre-labeled\\
        subject to desired data segmentation

    Args:
        dataframe (DataFrame): dataframe consisting of (cells * timepoints) x features, along with ts_id, frame, and label columns.
        label_col (str): column to treat as labels in dataframe
        segmentation (str, optional): Segmentation parameter. Defaults to "".
        label_val (int, optional): value for a positive hit (death). Defaults to 1.

    Returns:
        array: transition matrix for true hidden states, estimated from data and segmentation
    """    
    subframe = dataframe.copy()

    if segmentation != "":
        subframe = dataframe.query(segmentation)
    #now, we need to generate a count of which cells have death events.
    labels = [label_val in np.array(subframe[subframe[id_col] == x][label_col]) for x in list(set(subframe[id_col]))]
    p = np.sum(labels)/subframe.shape[0] #probability of living->death transition per frame
    return [[1-p,p],[0,1]]

def generate_condition_strings(data, columns):
    label_df = data[columns]
    distinct_vals = label_df[~label_df.duplicated()]
    condition_strings = [
    " and ".join([f"{col} == {repr(row[col])}" for col in columns])
    for _, row in distinct_vals.iterrows()]
    return condition_strings


"""simulated_data = pd.DataFrame()
simulated_data["concentration"] = np.append(np.ones(100)*50,np.ones(100)*80)
simulated_data["ts_id"] = np.concatenate((np.ones(40), np.ones(40)*2,np.ones(20)*3, np.ones(40)*4, np.ones(40)*5,np.ones(20)*6))
simulated_data["frame"] = np.concatenate((range(40), range(40),range(20), range(40), range(40),range(20)))
simulated_data["label"] = np.concatenate((np.zeros(30), np.ones(10), np.zeros(27), np.ones(13), np.zeros(55), np.ones(5), np.zeros(12), np.ones(28), np.zeros(9), np.ones(11)))



trans_mat_from_data(simulated_data, "label", "", 1)
trans_mat_from_data(simulated_data, "label", "concentration == 50", 1)
trans_mat_from_data(simulated_data, "label", "concentration == 80", 1)

[1 in simulated_data[simulated_data.ts_id == x]["label"] for x in list(set(simulated_data.ts_id))]
1 in np.array(simulated_data[simulated_data.ts_id == 2].label)


label_df = simulated_data[["label", "concentration"]]
label_df[~label_df.duplicated()]
generate_condition_strings(simulated_data, ["label", "concentration"])"""



#suppose we have a variety:

#generate a list of death times.  Before the split, use FP/TN and after the split use FN/TP.

# FP_list = np.linspace(0, 1, 26)
# FN_list = np.linspace(0, 1, 26)


# #generate data for set time length of 145
# t_max = 145
# percent_to_die = 0.5
# n_samples = 100
# samples = []
# samples_metrics = []
# tod_list = []
# for fp in FP_list:
#     for fn in FN_list:
#         tp = 1-fn
#         tn = 1-fp
#         samples_metrics.append([fp, fn])
#         tod_mini = []
#         samples_mini = []
#         for i in range(n_samples):
#             tod = t_max
#             if (np.random.rand() > percent_to_die):
#                 tod = int((np.random.rand()*t_max))
#             tod_mini.append(tod)
#             ans = np.concatenate((np.random.binomial(n = 1, p = fp, size = tod), np.random.binomial(n = 1, p = tp, size = (t_max - tod))))
#             samples_mini.append(ans)
#         samples.append(samples_mini)
#         tod_list.append(tod_mini)

# data = pd.DataFrame()
# data["fp"] = np.zeros(len(samples))
# data["fn"] = np.zeros(len(samples))
# data["tp"] = np.zeros(len(samples))
# data["tn"] = np.zeros(len(samples))
# data["abs_mean_dev"] = np.zeros(len(samples))
# data["RMSE"] = np.zeros(len(samples))
# data["median_abs_dev"] = np.zeros(len(samples))
# data["dev_sd"] = np.zeros(len(samples))
# #data["abs_mean_dev_trunc"] = np.zeros(len(samples))
# #data["RMSE_trunc"] = np.zeros(len(samples))
# #data["median_abs_dev_trunc"] = np.zeros(len(samples))
# #data["dev_sd_trunc"] = np.zeros(len(samples))

# #now we do inference:
# tod_list = np.array(tod_list)

# for i in range(len(samples_metrics)):
#     fp = samples_metrics[i][0]
#     fn = samples_metrics[i][1]
#     curr_samples = samples[i]
#     label_list = tod_list[i]
#     e_mat = [[1 - fp, fp],[fn, 1 - fn]]
#     t_mat = [[.995, .005],[0,1]] #known
#     full_results,scores = HMM_correction(curr_samples, t_mat, e_mat, n_components = 2)
#     pred_tod = np.array([t_max - len(np.flatnonzero(full_results[j])) for j in range(len(full_results))])
#     abs_mean_dev = np.mean(np.abs(pred_tod - tod_list[i]))
#     rmse = np.sqrt(np.mean((pred_tod - tod_list[i])**2))
#     med_abs_dev = np.median(np.abs(pred_tod - tod_list[i]))
#     dev_sd = np.std(pred_tod - tod_list[i])

#     data.loc[i, :] = [fp, fn, 1-fn, 1-fp,abs_mean_dev, rmse, med_abs_dev, dev_sd]

# import seaborn as sns
# import matplotlib.pyplot as plt
# data = data.sort_values(['fp','fn'],ascending = [True,True])
# df = data.pivot(index='fp', columns='fn', values='abs_mean_dev')
# hm = sns.heatmap(df, annot = True, fmt = ".1g")
# hm.invert_yaxis()
# plt.title("Classifier confusion vs. Mean Absolute Deviation in TOD")
# plt.show()