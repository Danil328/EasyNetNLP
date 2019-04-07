import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, auc


def PlotConfusionMatrix(y_test, pred, y_test_legit, y_test_fraud):
	cfn_matrix = confusion_matrix(y_test, pred)
	cfn_norm_matrix = np.array([[1.0 / y_test_legit, 1.0 / y_test_legit], [1.0 / y_test_fraud, 1.0 / y_test_fraud]])
	norm_cfn_matrix = cfn_matrix * cfn_norm_matrix

	fig = plt.figure(figsize=(15, 5))
	ax = fig.add_subplot(1, 2, 1)
	sns.heatmap(cfn_matrix, cmap='coolwarm_r', linewidths=0.5, annot=True, ax=ax)
	plt.title('Confusion Matrix')
	plt.ylabel('Real Classes')
	plt.xlabel('Predicted Classes')

	ax = fig.add_subplot(1, 2, 2)
	sns.heatmap(norm_cfn_matrix, cmap='coolwarm_r', linewidths=0.5, annot=True, ax=ax)

	plt.title('Normalized Confusion Matrix')
	plt.ylabel('Real Classes')
	plt.xlabel('Predicted Classes')
	plt.show(block=False)

	print('---Classification Report---')
	print(classification_report(y_test, pred))


def calculate_err(y_true, y_pred):
	fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
	fnr = 1 - tpr
	# print('FAR - {}'.format(fpr))
	# print('FRR - {}'.format(fnr))
	eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
	thresh = interp1d(fpr, thresholds)(eer)
	return eer, thresh
def calculate_score(y_true, y_pred, show=False):
	roc_score = roc_auc_score(y_true, y_pred)
	print(f'ROC_AUC score = {roc_score}')

	f1 = f1_score(y_true, np.round(y_pred))
	print(f'F1 score = {f1}')

	eer, thres = calculate_err(y_true, y_pred)
	print(f'EER score - {eer}')

	average_precision = average_precision_score(y_true, y_pred)

	print('Average precision score: {0:0.2f}'.format(average_precision))

	precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
	print(f"AUC PR {auc(recall, precision)}")

	if show:
		# PRECISION RECALL CURVE
		plt.figure()
		plt.title('PRECISION RECALL CURVE')
		plt.plot([0, 1], [0.5, 0.5], linestyle='--')
		plt.plot(recall, precision)

		# ROC AUC CURVE
		plt.figure()
		plt.title('ROC AUC CURVE')
		fpr, tpr, thresholds = roc_curve(y_true, y_pred)
		plt.plot([0, 1], [0, 1], linestyle='--')
		plt.plot(fpr, tpr)
		plt.show()

		y_test_legit = -(y_true - 1).sum()
		y_test_fraud = int(y_true.sum())
		PlotConfusionMatrix(y_true, np.round(y_pred), y_test_legit, y_test_fraud)

	return roc_score, f1, eer, auc(recall, precision)