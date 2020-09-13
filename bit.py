from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# 将数据中的特征和标签分离开 target代表是标签，data代表数据
from sklearn.metrics import recall_score
def data_process():
    file = open("F:\编程存放\毕业\data\\tcga_rna_sig.txt", "r", encoding="utf8")
    sub = file.readlines()
    target = []
    data = []
    for i in sub:
        data1 = []
        i = i.strip()
        target1 = i.split("	")[1]
        for x in i.split("	")[2:]:
            data1.append(float(x))
        target.append(target1)
        data.append(data1)
    return data,target
data,target=data_process()
file_data=open("F:\编程存放\毕业\data\s_data.txt","w",encoding="utf8")
file_target=open("F:\编程存放\毕业\data\s_target.txt","w",encoding="utf8")
def get_class_num(y_test):
    dict={}
    for i in y_test:
        if i in dict:
            dict[i]+=1
        else:
            dict[i]=1
    return dict
def get_weighted(y_test2,y_pred):
    weighted = 0
    weight_dict = {}
    true_dict = get_class_num(y_test2)
    for i in true_dict:
        weight_dict[i] = true_dict[i] / len(y_test2)
    pred_dict = get_class_num(y_pred)
    pred_class_dict = {}
    for i in range(len(y_pred)):
        if y_pred[i] == y_test2[i]:
            if y_pred[i] in pred_class_dict:
                pred_class_dict[y_pred[i]] += 1
            else:
                pred_class_dict[y_pred[i]] = 1
    for i in pred_dict:
        weighted += weight_dict[i] * (pred_class_dict[i] / pred_dict[i])
    return weighted
#将标签和数据分别写入到文件当中
for i in range(len(data)):
    for j in range(len(data[0])):
        file_data.write(str(data[i][j]))
        file_data.write(" ")
    file_data.write("\n")
for x in target:
    file_target.write(str(x))
    file_target.write("\n")
A_train_data=[]
A_train_target=[]
A_test_data=[]
A_test_target=[]
B_train_data=[]
B_train_target=[]
B_test_data=[]
B_test_target=[]
E_train_data=[]
E_train_target=[]
E_test_data=[]
E_test_target=[]
D_train_data=[]
D_train_target=[]
D_test_data=[]
D_test_target=[]
new_target=[]
pre=[]
test=[]
# 将原始数据集分为"A"和"B"两个了类别
for i in range(len(target)):
    if i<5007:
        new_target.append("A")
    else:
        new_target.append("B")
x_train, x_test, y_train, y_test= train_test_split(data, new_target, test_size=0.19,random_state=3)
X_train, X_test, Y_train, Y_test = train_test_split( data,target, test_size=0.19, random_state=3)

length=len(Y_test)
# 第一次分类在训练集上做训练，使其可以区分A和B
forest1=RandomForestClassifier()
forest1.fit(x_train,y_train)
y1_pred=forest1.predict(x_train)
# 在训练集上做测试，用于划分下一个决策树的训练集
for i in range(len(y1_pred)):
    if y1_pred[i]=="A":
        A_train_data.append(x_train[i])
        A_train_target.append(Y_train[i])
    else:
        B_train_data.append(x_train[i])
        B_train_target.append(Y_train[i])
# 在测试集上做预测，用于划分下一个决策器的测试集
y2_pred = forest1.predict(x_test)
for i in range(len(y2_pred)):
    if y2_pred[i]=="A":
        A_test_data.append(x_test[i])
        A_test_target.append(Y_test[i])
    else:
        B_test_data.append(x_test[i])
        B_test_target.append(Y_test[i])
#构造第二个决策器，训练数据为：A_train_data（为第一次预测的训练集结果）
decision2_tree=RandomForestClassifier()
decision2_tree.fit(A_train_data,A_train_target)
# yA_pred是真实类别C1,C2,C3的预测
yA_pred = decision2_tree.predict(A_test_data)
pre=pre+list(yA_pred)
test=test+list(A_test_target)
#对B_train_target进行标签重新替换
lable=[]
for x in B_train_target:
    if x=="C3":
        lable.append(x)
    else:
        lable.append("D")
decision3_tree=RandomForestClassifier()
decision3_tree.fit(B_train_data,lable)
yB_pred=decision3_tree.predict(B_train_data)
for c in range(len(yB_pred)):
    if yB_pred[c]=="D":
        D_train_data.append(B_train_data[c])
        D_train_target.append(B_train_target[c])
yC_pred=decision3_tree.predict(B_test_data)
for d in range(len(yC_pred)):
    if yC_pred[d]=="D":
        D_test_data.append(B_test_data[d])
        D_test_target.append(B_test_target[d])
prec=[]
testc=[]
for i in range(len(yC_pred)):
    if yC_pred[i]=="C3":
        prec.append(yC_pred[i])
        testc.append(B_test_target[i])
pre=pre+list(prec)
test=test+list(testc)
forest2=RandomForestClassifier()
lable2=[]
for x in D_train_target:
    if x=="C4":
        lable2.append(x)
    else:
        lable2.append("E")
forest2.fit(D_train_data,lable2)
yE_pred=forest2.predict(D_train_data)
for  i in range(len(yE_pred)):
    if yE_pred[i]=="E":
        E_train_data.append(D_train_data[i])
        E_train_target.append(D_train_target[i])
yF_pred=forest2.predict(D_test_data)

for  i in range(len(yF_pred)):
    if yF_pred[i]=="E":
        E_test_data.append(D_test_data[i])
        E_test_target.append(D_test_target[i])
pree=[]
teste=[]
for i in range(len(yF_pred)):
    if yF_pred[i]=="C4":
        pree.append(yF_pred[i])
        teste.append(D_test_target[i])
pre=pre+list(pree)
test=test+list(teste)
forest3=RandomForestClassifier()
forest3.fit(E_train_data,E_train_target)
yG_pred=forest3.predict(E_test_data)
# for i in range(len(yG_pred)):
#     if yG_pred[i] == E_test_target[i]:
#         all += 1
pre=pre+list(yG_pred)
test=test+list(E_test_target)
micro = recall_score(test, pre, average='micro')
print("micro:  ", micro)
macro = recall_score(test, pre, average='macro')
print("macro:  ", macro)
weighted = get_weighted(test, pre)
print("weighted:  ", weighted)
