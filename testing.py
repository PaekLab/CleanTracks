import os
os.chdir("./cleantracks")
import numpy as np
import cv2
import death_nn
from death_nn import em_mat_weights
import data_utilities
import death_labeling
import pandas as pd
import keras
import tensorflow
import sklearn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, multilabel_confusion_matrix
import matplotlib.pyplot as plt

train_X = pd.read_csv("./data/train_X.csv")
test_X = pd.read_csv("./data/test_X.csv")
train_Y = np.array(pd.read_csv("./data/train_Y.csv", header=None))
test_Y = np.array(pd.read_csv("./data/test_Y.csv", header=None))
model = keras.models.load_model("./models/death_frame_predictor.keras")
cols_to_drop = ["ts_id", "conc", "frame"]
rtrain_X = train_X.drop(cols_to_drop, axis = 1)
rtrain_X = rtrain_X.iloc[:,1:]

rtest_X = test_X.drop(cols_to_drop, axis=1)
rtest_X = rtest_X.iloc[:,1:]

train_results = model.predict(rtrain_X)
test_results = model.predict(rtest_X)
np.sum((train_results.round() == train_Y).flatten())/len(train_Y)
np.sum((test_results.round() == test_Y).flatten())/len(test_Y)
cm = confusion_matrix(test_Y, test_results.round())
np.mean(np.abs((train_results.round() - train_Y)))

g = ConfusionMatrixDisplay(cm)
g.plot()
plt.show()



# Calculate the correlation matrix
#correlation matrix of rtrain_X
cov_mat = rtest_X.corr()
plt.imshow(cov_mat, cmap='viridis', interpolation='nearest')
plt.colorbar()
#axis labels
plt.xticks(ticks=np.arange(rtrain_X.shape[1]), labels=rtrain_X.columns, rotation=90)
plt.yticks(ticks=np.arange(rtrain_X.shape[1]), labels=rtrain_X.columns)
plt.xlabel("Features")
plt.ylabel("Features")
plt.title("Correlation Matrix of rtrain_X")
plt.show()




model_test, args = death_nn.early_stopping_ffnn(rtrain_X.shape[-1], train_Y.shape[-1])

model_test.fit(rtrain_X, train_Y, verbose = 0, epochs = 100)
train_results = model_test.predict(rtrain_X)
test_results = model_test.predict(rtest_X)
np.sum((train_results.round() == train_Y).flatten())/len(train_Y)
np.sum((test_results.round() == test_Y).flatten())/len(test_Y)
cm = confusion_matrix(test_Y, test_results.round())
np.mean(np.abs((train_results.round() - train_Y)))

g = ConfusionMatrixDisplay(cm)
g.plot()
plt.show()

train_X["death_pred"] = train_results.round()
test_X["death_pred"] = test_results.round()
train_X["death_gt"] = train_Y
test_X["death_gt"] = test_Y

#determine stationarity
#generate false negative/false positive rate per frame
frame_list = list(set(test_X.frame))
frame_list.sort()
fn_per_frame = []
fp_per_frame = []
for f in frame_list:
    curr_frame = test_X[test_X.frame == f]
    cm = em_mat_weights(curr_frame.death_gt, curr_frame.death_pred)
    fn_per_frame.append(cm[1,0])
    fp_per_frame.append(cm[0,1])

np.array(fn_per_frame)
np.array(fp_per_frame)
plt.plot(frame_list, fn_per_frame)
plt.plot(frame_list, fp_per_frame)
plt.show()


total_data = pd.concat([train_X, test_X])

t_mat = death_labeling.trans_mat_from_data(total_data, "death_pred")
t_mat
e_mat = em_mat_weights(test_results, test_Y)
e_mat


## Testing for auto-tracked data using pre-trained model on hand-tracked data


## Testing for hand-tracked data using model trained on auto-tracked data
#read in data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data_trimmed = pd.read_csv("./data/all_tracks_dataframe.csv")
data_trimmed = data_trimmed.iloc[:, 1:data_trimmed.shape[1]]
data_trimmed["gt_death"] = 0


death_data = pd.read_csv("./data/death_track_df_hand_labeled.csv")
death_data = death_data.iloc[:,1:-7]
data_death_labeled = pd.read_csv("./death_labeled_mini.csv")
data_training = data_trimmed.copy()

good_pos = data_death_labeled.loc[:,["position", "track"]]

new_data_arr = [data_training[((data_training.position == good_pos.iloc[i,0]) & (data_training.track_id == good_pos.iloc[i,1]))] for i in range(len(good_pos))]


for i in range(len(data_death_labeled)):
    curr_death_data = data_death_labeled.iloc[i,:]
    if (curr_death_data.tod == -1):
        new_data_arr[i].loc[:,"series_len"] = curr_death_data["series_length"]
        continue
    else:
        new_data_arr[i].loc[:,"series_len"] = curr_death_data["series_length"]
        new_data_arr[i].loc[new_data_arr[i].frame_id >= (curr_death_data.tod + np.min(new_data_arr[i].frame_id)),["gt_death"]] = 1

bad_tracks_removed = [new_data_arr[i] for i in range(len(new_data_arr)) if np.sum(new_data_arr[i].area <=100) == 0]
data_training = pd.concat(new_data_arr, axis = 0)
data_training = pd.concat(bad_tracks_removed, axis = 0)

experiment_labels = ["UT", "50", "80", "100", "80J14","100SFRX"]
data_training["experiment_label"] = ""
data_training.loc[data_training.position < 10,"experiment_label"] = "UT"
data_training.loc[data_training.position.isin([10,11]),"experiment_label"] = "50"
data_training.loc[data_training.position.isin([30,31]),"experiment_label"] = "80"
data_training.loc[data_training.position == 32,"experiment_label"] = "100"
data_training.loc[data_training.position.isin([54,55]),"experiment_label"] = "80J14"
data_training.loc[data_training.position == 111,"experiment_label"] = "100SFRX"

data_training["id_col"] = "pos" + data_training["position"].astype(str) + "track" + data_training["track_id"].astype(str)



data_subset_pos = data_training[(data_training.gt_death == 0) ]#& (data_training.series_len == 144)]
data_subset_neg = data_training[(data_training.gt_death == 1) ]#& (data_training.series_len == 144)]

data_for_training = pd.concat([data_subset_pos.sample(n = 15000, random_state = 22), data_subset_neg.sample(n = 20000, random_state = 22)])


X_train, X_test, y_train, y_test = train_test_split(data_for_training, data_for_training.gt_death, test_size = .2, train_size = .8, random_state = 4)
subset_cols = ["area","radius", "radius_delta","area_delta","vel", "solidity", "solidity_delta"]
train_input = X_train.loc[:,subset_cols]
test_input = X_test.loc[:,subset_cols]
train_indices = X_train.index
test_indices = X_test.index



scaler = StandardScaler()
X_scaled = scaler.fit_transform(train_input)
test_scaled = scaler.transform(test_input)
y_train = np.array(y_train).reshape((len(y_train), 1))
y_test = np.array(y_test).reshape((len(y_test), 1))


model_auto, args = death_nn.early_stopping_ffnn(train_input.shape[-1], y_train.shape[-1])
model_auto.fit(X_scaled, y_train, verbose = 0, epochs = 800, callbacks = args)
model_auto.save("./models/death_frame_predictor_auto_20min.keras")
model_auto.load_weights("./models/death_frame_predictor_auto_20min.keras")
pred = model_auto.predict(X_scaled)
cm = confusion_matrix(y_train, pred.round())
cm
acc = np.sum(y_train == np.round(pred))/len(y_train)
acc

pred = model_auto.predict(test_scaled)
cm = confusion_matrix(y_test, pred.round())
cm
acc = np.sum(y_test == np.round(pred))/len(y_test)
acc

e_mat = em_mat_weights(np.round(pred), y_test)
data_test_all = data_training.loc[np.array(test_indices),:]
data_in_UT = data_test_all.loc[data_test_all.experiment_label == "UT",subset_cols]
pred_UT = model_auto.predict(scaler.transform(data_in_UT))
cm_ut = confusion_matrix(y_test[data_test_all.experiment_label == "UT"], np.round(pred_UT), labels = [0,1])
cm_round_ut = cm_ut/np.sum(cm_ut, axis=1, keepdims=True)
cm_round_ut[1,0] = 0
cm_round_ut[1,1] = 0
e_mat = em_mat_weights(np.round(pred_UT),y_test[data_test_all.experiment_label == "UT"])
#UT -- 79/21
data_in_50 = data_test_all.loc[data_test_all.experiment_label == "50",subset_cols]
pred_50 = model_auto.predict(scaler.transform(data_in_50))
cm_50 = confusion_matrix(y_test[data_test_all.experiment_label == "50"], np.round(pred_50))
cm_round_50 = cm_50/np.sum(cm_50, axis=1, keepdims=True)
#50 -- 86/14 | 51/49 -- too few samples
data_in_80 = data_test_all.loc[data_test_all.experiment_label == "80",subset_cols]
pred_80 = model_auto.predict(scaler.transform(data_in_80))
cm_80 = confusion_matrix(y_test[data_test_all.experiment_label == "80"], np.round(pred_80))
cm_round_80 = cm_80/np.sum(cm_80, axis=1, keepdims=True)

#50 -- 86/14 | 51/49 -- too few samples
experiment_labels = ["UT", "50", "80", "100", "80J14","100SFRX"]

data_in_100 = data_test_all.loc[data_test_all.experiment_label == "100",subset_cols]
pred_100 = model_auto.predict(scaler.transform(data_in_100))
cm_100 = confusion_matrix(y_test[data_test_all.experiment_label == "100"], np.round(pred_100))
cm_round_100 = cm_100/np.sum(cm_100, axis=1, keepdims=True)

data_in_100s = data_test_all.loc[data_test_all.experiment_label == "100SFRX",subset_cols]
pred_100s = model_auto.predict(scaler.transform(data_in_100s))
cm_100s = confusion_matrix(y_test[data_test_all.experiment_label == "100SFRX"], np.round(pred_100s))
cm_round_100s = cm_100s/np.sum(cm_100s, axis=1, keepdims=True)

data_in_80j = data_test_all.loc[data_test_all.experiment_label == "80J14",subset_cols]
pred_80j = model_auto.predict(scaler.transform(data_in_80j))
cm_80j = confusion_matrix(y_test[data_test_all.experiment_label == "80J14"], np.round(pred_80j))
cm_round_80j = cm_80j/np.sum(cm_80j, axis=1, keepdims=True)

# train the model on each experiment separately and generate confusion matrices
def train_experiment_model(data, subset_cols, random_seed=42):
    X = data.loc[:, subset_cols]
    y = np.array(data.gt_death).reshape((len(data.gt_death), 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model, args = death_nn.early_stopping_ffnn(X_train.shape[-1], y_train.shape[-1])
    model.fit(X_train_scaled, y_train, verbose=0, epochs=800, callbacks=args)
    pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, pred.round())
    return cm

cm_ut = train_experiment_model(data_training[data_training.experiment_label == "UT"], subset_cols)
cm_50 = train_experiment_model(data_training[data_training.experiment_label == "50"], subset_cols)
cm_80 = train_experiment_model(data_training[data_training.experiment_label == "80"], subset_cols)
cm_100 = train_experiment_model(data_training[data_training.experiment_label == "100"], subset_cols)
cm_100s = train_experiment_model(data_training[data_training.experiment_label == "100SFRX"], subset_cols)
cm_80j = train_experiment_model(data_training[data_training.experiment_label == "80J14"], subset_cols)

#plot all scaled confusion matrices
import seaborn as sns
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, title):
    cm_scaled = cm / np.sum(cm, axis=1, keepdims=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_scaled, annot=True, fmt=".2f", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(ticks=[0.5, 1.5], labels=["No Death", "Death"])
    plt.yticks(ticks=[0.5, 1.5], labels=["No Death", "Death"], rotation=0)
    plt.show()
plot_confusion_matrix(cm_ut, "UT Experiment Confusion Matrix")
plot_confusion_matrix(cm_50, "50 Experiment Confusion Matrix")
plot_confusion_matrix(cm_80, "80 Experiment Confusion Matrix")
plot_confusion_matrix(cm_100, "100 Experiment Confusion Matrix")
plot_confusion_matrix(cm_100s, "100SFRX Experiment Confusion Matrix")
plot_confusion_matrix(cm_80j, "80J14 Experiment Confusion Matrix")



data_scaled = scaler.transform(data_training.loc[:,subset_cols])
pred = model_auto.predict(data_scaled)
data_y = np.array(data_training.gt_death).reshape((len(data_training.gt_death), 1))
cm = confusion_matrix(data_y, np.round(pred))
cm

acc = np.sum(data_y == np.round(pred))/len(data_y)
acc


#alternate version:
data_training["death_predict"] = np.round(pred)
frame_list = list(set(data_training.frame_id))
frame_list.sort()
fn_per_frame = []
fp_per_frame = []
for f in frame_list:
    curr_frame = data_training[data_training.frame_id == f]
    cm = em_mat_weights(curr_frame.gt_death, curr_frame.death_predict)
    fn_per_frame.append(cm[1, 0])
    fp_per_frame.append(cm[0, 1])

fn_per_frame = np.array(fn_per_frame)
fp_per_frame = np.array(fp_per_frame)

# Smooth the data using a rolling mean

from scipy.ndimage import gaussian_filter1d
#rolling mean
import pandas as pd
fn_per_frame = pd.Series(fn_per_frame)
fp_per_frame = pd.Series(fp_per_frame)


fn_per_frame_smoothed = fn_per_frame.rolling(window=10, center=True).mean()
#fn_per_frame#gaussian_filter1d(fn_per_frame, sigma=2)
fp_per_frame_smoothed = fp_per_frame.rolling(window = 10, center = True).mean()#fp_per_frame#gaussian_filter1d(fp_per_frame, sigma=2)

# Calculate error bands (standard error)

fn_std = np.std(fn_per_frame)/np.sqrt(len(fn_per_frame))*1.96
fp_std = np.std(fp_per_frame)/np.sqrt(len(fp_per_frame))*1.96

end = -5
plt.plot(frame_list[1:end], fn_per_frame_smoothed[1:end], label='False Negative Rate', color='blue')
plt.fill_between(frame_list[1:end], 
                 fn_per_frame_smoothed[1:end] - fn_std, 
                 fn_per_frame_smoothed[1:end] + fn_std, 
                 color='blue', alpha=0.2, label=None)

plt.plot(frame_list[1:end], fp_per_frame_smoothed[1:end], label='False Positive Rate', color='orange')
plt.fill_between(frame_list[1:end], 
                 fp_per_frame_smoothed[1:end] - fp_std, 
                 fp_per_frame_smoothed[1:end] + fp_std, 
                 color='orange', alpha=0.2, label=None)
#add means as dashed lines
plt.axhline(y=np.mean(fn_per_frame_smoothed[1:end]), color='blue', linestyle='--', label=None)
plt.axhline(y=np.mean(fp_per_frame_smoothed[1:end]), color='orange', linestyle='--', label=None)
plt.xlabel('Frame', fontsize=20)
plt.ylabel('Rate', fontsize=20)
plt.title('False Negative Rate and False Positive Rate per Frame (95% CI, mean smoothed)', fontsize = 24)
#increase font size
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.tight_layout()
plt.show()




from statsmodels.tsa.stattools import adfuller
# Perform Augmented Dickey-Fuller test

result = adfuller(fn_per_frame_smoothed[1:-1], autolag = 'AIC')
result = adfuller(fp_per_frame_smoothed[1:-1], autolag = 'AIC')

#new data from chance#
data_trimmed = pd.read_csv("./data/all_tracks_dataframe_chance_unlabeled.csv")
data_trimmed = data_trimmed.iloc[:, 1:data_trimmed.shape[1]]
#handling interpolation:
interp_df = data_trimmed.copy()
interp_df = interp_df.sort_values(by= ["position", "track_id", "frame_id"])
position_list = list(interp_df.position.drop_duplicates())
position_list.sort()
interp_df["new_time"] = interp_df["frame_id"]*30
interp_df.area = interp_df["area"].astype(float)
interp_df.area_delta = interp_df["area_delta"].astype(float)
pos_df = pd.DataFrame()
for i in position_list:
    subset = interp_df[interp_df.position == i]
    track_list = list(subset.track_id)
    track_list.sort()
    for t in track_list:
        curr_vals = subset[subset.track_id == t]
        curr_vals_features = curr_vals[subset_cols]
        new_times, new_features = interpolate_features(np.array(curr_vals_features).T, original_spacing=30, target_spacing=20, method='linear', existing_timestamps=list(curr_vals.new_time))
        track_df = pd.DataFrame(new_features.T, columns=curr_vals_features.columns)
        track_df["position"] = i
        track_df["track_id"] = t
        track_df["new_times"] = np.arange(len(new_times))
        track_df["new_frame"] = track_df["new_times"]/30
    pos_df = pd.concat([pos_df, track_df], axis = 0) if 'pos_df' in locals() else track_df



pos_df.new_times = pos_df.new_times*20
pos_df.new_frame = pos_df.new_frame*30



data_t = data_training.loc[:,subset_cols]
chance_scaled = scaler.transform(data_t)
pred_death = model_auto.predict(chance_scaled)
##

data_training["new_pred"] = np.round(pred_death)
data_training["id_col"] = [("pos" + str(data_training.iloc[i,:].position) + "track" + str(data_training.iloc[i,:].track_id)) for i in range(len(data_training))]
data_training["frame_id"] = data_trimmed["frame_id"]
data_training["track_id"] = data_trimmed["track_id"]
data_training["position"] = data_trimmed["position"]
t_mat = [[0.95,0.05],[0,1]]
t_mat = death_labeling.trans_mat_from_data(data_training, "gt_death", id_col = "id_col")
t_mat

e_mat

e_mat_2 = em_mat_weights(data_y, np.round(pred))
e_mat_2 = em_mat_weights(np.round(pred), data_y)


pred_hm_new = np.zeros((len(list(set(data_training.id_col))), np.max(data_training.frame_id + 1)))
pred_hm = np.zeros((len(list(set(data_training.id_col))), np.max(data_training.frame_id + 1)))
gt_hm = np.zeros((len(list(set(data_training.id_col))), np.max(data_training.frame_id + 1)))
anno = []
key = list(set(data_training.id_col))
frame_lists = []
for ind, i in enumerate(list(set(data_training.id_col))):
    curr_data = data_training[data_training.id_col == i]
    anno.append(curr_data.position.iloc[0])
    frame_lists.append(curr_data.frame_id)
    for jnd, j in enumerate(curr_data.frame_id):
        pred_hm_new[ind, j] = curr_data[curr_data.frame_id == j].new_pred

        pred_hm[ind, j] = curr_data[curr_data.frame_id == j].death_predict
        gt_hm[ind, j] = curr_data[curr_data.frame_id == j].gt_death

#t_mat = [[.95, .0],[0, 1]]
e_mat = em_mat_weights(np.round(pred), y_test)
t_mat = death_labeling.trans_mat_from_data(data_training, "gt_death", id_col = "id_col")
results,scores = death_labeling.HMM_correction(pred_hm_new, t_mat, e_mat, n_components=2)
t2 = [[.999, .001],[0, 1]]
e2 = [[1, 0], [0,1]]
gt_lab,scoresgt = death_labeling.HMM_correction(gt_hm, t2, e2, n_components = 2)
rowsums = np.sum(gt_lab, axis = 1)
sort_order = np.argsort(rowsums)[::-1]#do a rolling mean on rows in pred_hm_new
#sort_order by scores
#sort_order = np.argsort(scores)
#pred_hm_new_mean = np.round(pd.DataFrame(pred_hm_new).rolling(window=3, center  = True, min_periods = 1).mean().values)
#anything before the last location of a 1 in each row should be set to 0, everything after set to 1.
import seaborn as sns
f, axarr = plt.subplots(1,3)

sns.heatmap(gt_lab[sort_order], ax=axarr[0], cbar=False)
sns.heatmap(pred_hm_new[sort_order], ax=axarr[1], cbar=False)
sns.heatmap(results[sort_order], ax=axarr[2], cbar=True)
#plt.title("Ground Truth vs. Predicted Framewise Death Labels (NN)")
#sns.heatmap(np.round(pred_hm_new_mean[sort_order]), ax=axarr[3], cbar=True)
#sns.heatmap(gt_hm[sort_order], ax=axarr[1,0], cmap="viridis", cbar=True)
plt.show()

###EXPERIMENT_UPDATE
def process_experiment_partition(data_training, experiment_label, e_mat):
    """
    Process data for a specific experimental partition and return gt_lab, pred_hm_new, and results.

    Parameters:
    data_training (DataFrame): The training data containing all experiments.
    experiment_label (str): The label of the experiment to subset the data.

    Returns:
    tuple: gt_lab, pred_hm_new, results
    """
    # Subset the data by the experiment label
    subset_data = data_training[data_training.experiment_label == experiment_label]

    # Initialize matrices
    pred_hm_new = np.zeros((len(list(set(subset_data.id_col))), np.max(subset_data.frame_id + 1)))
    pred_hm = np.zeros((len(list(set(subset_data.id_col))), np.max(subset_data.frame_id + 1)))
    gt_hm = np.zeros((len(list(set(subset_data.id_col))), np.max(subset_data.frame_id + 1)))
    anno = []
    key = list(set(subset_data.id_col))
    frame_lists = []

    # Populate matrices
    for ind, i in enumerate(key):
        curr_data = subset_data[subset_data.id_col == i]
        anno.append(curr_data.position.iloc[0])
        frame_lists.append(curr_data.frame_id)
        for jnd, j in enumerate(curr_data.frame_id):
            pred_hm_new[ind, j] = curr_data[curr_data.frame_id == j].new_pred
            pred_hm[ind, j] = curr_data[curr_data.frame_id == j].death_predict
            gt_hm[ind, j] = curr_data[curr_data.frame_id == j].gt_death

    # Transition and emission matrices
    e_mat = e_mat
    t_mat = [[.95, .05], [0, 1]]
    t_mat = death_labeling.trans_mat_from_data(subset_data, "gt_death", id_col="id_col")

    # HMM corrections
    results, scores =HMM_correction(pred_hm_new, t_mat, e_mat, n_components=2)
    t2 = [[.999, .001], [0, 1]]
    e2 = [[1, 0], [0, 1]]
    gt_lab, scoresgt = HMM_correction(gt_hm, t2, e2, n_components=2)

    return gt_lab, pred_hm_new, results, scores, scoresgt
results_all = {}
experiment_labels = ["UT", "50", "80", "100", "80J14", "100SFRX"]

gt_lab_50, pred_hm_new_50, results_50 = process_experiment_partition(data_training, "50", cm_round_50)    
rowsums = np.sum(gt_lab_50, axis = 1)
sort_order = np.argsort(rowsums)[::-1]
f, axarr = plt.subplots(1,3)

sns.heatmap(gt_lab_50[sort_order], ax=axarr[0], cbar=False)
sns.heatmap(pred_hm_new_50[sort_order], ax=axarr[1], cbar=False)
sns.heatmap(results_50[sort_order], ax=axarr[2], cbar=True)
#plt.title("Ground Truth vs. Predicted Framewise Death Labels (NN)")
#sns.heatmap(np.round(pred_hm_new_mean[sort_order]), ax=axarr[3], cbar=True)
#sns.heatmap(gt_hm[sort_order], ax=axarr[1,0], cmap="viridis", cbar=True)
plt.show()

gt_lab_80, pred_hm_new_80, results_80, scores_80, scores_gt80 = process_experiment_partition(data_training, "80", cm_round_80)    
rowsums = np.sum(gt_lab_80, axis = 1)
sort_order = np.argsort(rowsums)[::-1]
f, axarr = plt.subplots(1,3)

sns.heatmap(gt_lab_80[sort_order], ax=axarr[0], cbar=False)
sns.heatmap(pred_hm_new_80[sort_order], ax=axarr[1], cbar=False)
sns.heatmap(results_80[sort_order], ax=axarr[2], cbar=True)
#plt.title("Ground Truth vs. Predicted Framewise Death Labels (NN)")
#sns.heatmap(np.round(pred_hm_new_mean[sort_order]), ax=axarr[3], cbar=True)
#sns.heatmap(gt_hm[sort_order], ax=axarr[1,0], cmap="viridis", cbar=True)
plt.show()

gt_lab_100, pred_hm_new_100, results_100 = process_experiment_partition(data_training, "100", cm_round_100)    
rowsums = np.sum(gt_lab_100, axis = 1)
sort_order = np.argsort(rowsums)[::-1]
f, axarr = plt.subplots(1,3)

sns.heatmap(gt_lab_100[sort_order], ax=axarr[0], cbar=False)
sns.heatmap(pred_hm_new_100[sort_order], ax=axarr[1], cbar=False)
sns.heatmap(results_100[sort_order], ax=axarr[2], cbar=True)
#plt.title("Ground Truth vs. Predicted Framewise Death Labels (NN)")
#sns.heatmap(np.round(pred_hm_new_mean[sort_order]), ax=axarr[3], cbar=True)
#sns.heatmap(gt_hm[sort_order], ax=axarr[1,0], cmap="viridis", cbar=True)
plt.show()


#update_labels:
data_training["pred_corr"] = 0
for ind, i in enumerate(list(set(data_training.id_col))):
    frame_pos = data_training[data_training.id_col == i].frame_id
    data_training.loc[data_training.id_col == i, "pred_corr"] = results[ind, frame_pos]

data_trimmed["death_pred"] = data_training["pred_corr"]
data_trimmed.to_csv("./data/all_tracks_dataframe_chancemay_annot.csv", index=False)
data_training.to_csv("./data/all_tracks_dataframe_lisa_annot.csv", index=False)
t_max = 144
results_50t = results[np.array(scores) > -50]
gt_lab_50t =  gt_lab[np.array(scores) > -50]
pt = np.array([t_max - len(np.flatnonzero(results_50t[j])) for j in range(len(results_50t))])
gt = np.array([t_max - len(np.flatnonzero(gt_lab_50t[j])) for j in range(len(gt_lab_50t))])
#mt = np.array([t_max - len(np.flatnonzero(pred_hm_new_mean[j])) for j in range(len(pred_hm_new_mean))])

mean_abs_diff = np.mean(np.abs(pt - gt))
mean_abs_diff
plt.boxplot(np.abs(pt - gt))
#density plot

plt.hist(np.abs(pt - gt), density=True, bins = 50, alpha = 0.70, color = 'red')
sns.kdeplot(np.abs(pt - gt), fill = True)
plt.xlabel("Absolute Difference between Predicted and Ground Truth TOD")
plt.show()
# gt_lab[sort_order]
# gt_col = death_data_labeled.tod
# gt_col[gt_col == -1] = 144

# anno = np.array(anno)
# labels = pd.read_csv("./data/labels.csv")
# anno_labels = np.array([labels.loc[i//5, "experiment_label"] for i in anno])
##new heatmap##

# # Create dummy heatmap data
# import distinctipy

# # Create row annotation vector (can be categorical or continuous)
# row_annotation = anno_labels

# data_trimmed["id_col"] = data_training["id_col"]
# data_trimmed["death_val"] = 0
# data_trimmed["experiment_label"] = ""
# for ind, val in enumerate(key):
#     data_trimmed.loc[data_trimmed.id_col == val, "death_val"] = results[ind]
#     data_trimmed.loc[data_trimmed.id_col == val, "experiment_label"] = row_annotation[ind]

# distinct_list = list(set(row_annotation))
# distinct_list = list(labels.experiment_label)
# #colors = distinctipy.get_colors(len(distinct_list))
# lut = dict(zip(distinct_list, sns.hls_palette(len(distinct_list), l=0.5, s=0.8)))
# # Map the annotation to colors
# #lut = {'A': 'red', 'B': 'green', 'C': 'blue'}
# row_colors = pd.Series(row_annotation).map(lut)
# row_colors = row_colors.iloc[sort_order]

# # Plot clustermap with row_colors
# g = sns.clustermap(results[sort_order], row_cluster=False, col_cluster=False, row_colors=[row_colors])

# # Add legend for row annotation
# for label in lut:
#     g.ax_row_colors.bar(0, 0, color=lut[label], label=label, linewidth=0)
# g.ax_row_colors.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Group")
# plt.tight_layout()
# plt.show()


#-------#
#---death checking -- 25-150 uM h2o2----#


#-------#




# import numpy as np
# import umap
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler

# def umap_cluster_plot(data, n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', labels=None):
#     """
#     Perform UMAP dimensionality reduction and plot the clusters.
    
#     Parameters:
#     data (array-like): High-dimensional input data.
#     n_components (int): Number of dimensions for UMAP projection.
#     n_neighbors (int): Number of neighbors considered for UMAP.
#     min_dist (float): Minimum distance between embedded points.
#     metric (str): Distance metric used by UMAP.
#     labels (array-like, optional): Cluster labels for coloring the plot.
#     """
    
#     # Standardize data
#     scaled_data = StandardScaler().fit_transform(data)
    
#     # Apply UMAP
#     reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
#     embedding = reducer.fit_transform(scaled_data)
    
#     # Plot results
#     plt.figure(figsize=(8, 6))
#     scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=10, alpha=0.7)
    
#     if labels is not None:
#         plt.colorbar(scatter, label="Cluster Label")
    
#     plt.xlabel("UMAP 1")
#     plt.ylabel("UMAP 2")
#     plt.title("UMAP Clustering Visualization")
#     plt.show()
    
#     return embedding

# data_training_umap = data_training[subset_cols]


# import seaborn as sns
# sns.pairplot(data_training_umap, hue='gt_death', height = 1)
# plt.show()
# umap_cluster_plot(data_training_umap, labels = data_training.gt_death)



# #infer an HMM with two hidden states using data_training[subset_cols] as the observations
# from hmmlearn import hmm
# # Define the model with length 144
# model_hmm = hmm.GaussianHMM(n_components=2, covariance_type='diag', n_iter=1000, random_state=42)
# # Fit the model to the data
# #X_train has time series data in long format, with each row being a time step and each column being a feature, with distinct samples as tracks
# X_train_data = data_training.loc[(data_training.series_len == 144) & (data_training.experiment_label == "UT"),:]
# X_train_data_in = X_train_data[subset_cols]
# model_hmm.fit(X_train_data_in, lengths=X_train_data_in.groupby(X_train_data.id_col).size().values)
# # Predict the hidden states
# hidden_states, prob = model_hmm.decode(X_train_data_in)