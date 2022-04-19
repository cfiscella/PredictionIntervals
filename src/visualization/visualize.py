import matplotlib.pyplot as plt

baseline_to_read = open(,"rb")
baseline_result_dict = pickle.load(baseline_to_read)

model_to_read = open(,"rb")
model_result_dict = pickle.load(model_to_read)

truth_to_read = open(,"rb")
truth_dict = pickle.load(truth_to_read)

for etf in model_result_dict.keys():
  for experiment in model_result_dict[etf].keys():

    
    model_lower = pd.Series(model_result_dict[etf][experiment][:,0],name = 'lower')
    model_upper = pd.Series(model_result_dict[etf][experiment][:,1],name = 'upper')

    baseline_lower = pd.Series(baseline_result_dict[etf][experiment][:,0],name = 'lower')
    baseline_upper = pd.Series(baseline_result_dict[etf][experiment][:,1],name = 'upper')

    true = pd.DataFrame(truth_dict[etf][experiment].flatten(),columns = truth_index)

    plot_model_title = "{} Out of Sample Results: Experiment {}".format(etf,experiment)

    plt.figure()
    plt.plot(model_lower.values,color = 'red')
    plt.plot(model_upper.values,color = 'red')
    plt.plot(baseline_lower[-len(model_lower):].values,color = 'gray',alpha = .2)
    plt.plot(baseline_upper[-len(model_lower):].values,color = 'gray',alpha = .2)
    plt.plot(true[-len(model_lower):].values,color = 'blue')

    plt.title(plot_model_title)
    plt.save_fig("reports/figures/"+etf+"/"+etf+"_"+experiment)

