import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sn

class neuralnetwork:
    def __init__(self,file_name,feature_list,classes_list,bias_flag, n_hidden,nepoch,lr,n_neurals,functions):
        self.data = pd.read_csv(file_name)
        self.encodelist = {}
        self.feature_list = feature_list
        self.classes_list = classes_list
        self.BF = bias_flag
        self.num_epoch = int(nepoch)
        self.LR = float(lr)
        self.n_hidden = n_hidden
        self.n_neurals = n_neurals
        self.train_data = []
        self.test_data = []
        self.expected = []
        self.functions=functions
        self.network=[]
        self.weights_list=[]
        self.X0 = 0
        if self.BF == 1 or self.BF == True:
            self.X0 = 1

    def sigmoid(self, net):
        return 1 / (1 + (np.exp(-net)))

    def Hyperbolic_tangent(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def get_train_test(self):
        classflag = [True if x in self.classes_list else False for x in self.data["Class"]]
        feature_data = self.data[classflag]
        feature_data = feature_data[self.feature_list + ["Class"]]
        self.train_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        feature_data = shuffle(feature_data)
        for iris_class in feature_data["Class"].unique():
            features = feature_data[feature_data["Class"] == iris_class]
            print(len(features))
            train_split = features[:30]
            test_split = features[30:]
            self.train_data = self.train_data.append(train_split)
            self.test_data = self.test_data.append(test_split)
        self.train_data=shuffle(self.train_data)
        self.test_data=shuffle(self.test_data)
        self.encode_list = dict(zip(self.classes_list, [1, 0, -1]))
        self.train_data["X0"] = [self.X0 for x in self.train_data[self.feature_list[0]]]
        self.test_data["X0"] = [self.X0 for x in self.test_data[self.feature_list[0]]]
        self.train_data['Class'] = [self.encode_list[clas] for clas in self.train_data['Class']]
        self.test_data['Class'] = [self.encode_list[clas] for clas in self.test_data['Class']]
        self.train_data=self.train_data[["X0"]+self.feature_list+["Class"]]
        self.test_data = self.test_data[["X0"] + self.feature_list + ["Class"]]



    def compile(self):
        max=0
        weights=[]
        self.initialize_network()
        self.get_train_test()

        exp = [1, 0, -1]
        totalerror=0
        for e in range(self.num_epoch):
            error = 0
            true=0
            for row in self.train_data.values:
                self.forward(row[:-1])
                self.Back(row[-1])
                self.Update_weights()

                if row[-1]!=exp[np.argmax(self.Xs[-1][0])]:
                    error+=1
                else:
                    true+=1

                self.Xs=[]
            Error=int((error / len(self.train_data.values))*100)
            ACC=int((true / len(self.train_data.values))*100)
            if ACC>=max:
                weights=self.weights_list
                max=ACC
            self.weights_list=weights
            print("Train epoch" + str(e) + ":" + "accuracy=" + str(ACC) + "%  Error: " + str(Error) )


        self.test()

    def initialize_network(self):
        self.weights_list=[np.random.random((self.n_neurals[i] + 1,self.n_neurals[i + 1])) for i in range(self.n_hidden+1)]
        self.Xs=[]
        self.delta=[]
        self.netvalue=[]


    def forward(self,row):
        #Input Layer Data

        row = np.expand_dims(row, axis=1)
        data_input=np.transpose(row)
        self.Xs.append(data_input)
        XS = np.transpose(row)

        for w in range(len(self.weights_list)):
        # feed forward
            WX = np.dot(XS, self.weights_list[w])
            self.netvalue.append(WX)
            if self.functions=="Sigmoind":
                act_WX = [self.sigmoid(dp) for dp in WX]
            else:
                act_WX = [self.Hyperbolic_tangent(dp) for dp in WX]
            self.Xs.append(act_WX)
            XS= np.insert(act_WX, 0, self.X0, axis=1)


    def Back(self,expected):
        expected_output=[]
        if expected == 1:
            expected_output = np.array([1, 0, 0])
        elif expected == 0:
            expected_output = np.array([0, 1, 0])
        elif expected == -1:
            expected_output = np.array([0, 0, 1])
        self.get_delta(expected_output)

    def get_delta(self,expected_output):
        last_layer_error = (expected_output - self.Xs[-1])
        derivative_value=[]
        if self.functions == "Sigmoind":
            derivative_value = np.array(self.Xs[-1]) * (1 - np.array(self.Xs[-1]))
        else:
            derivative_value = np.array((1 - (np.array(self.Xs[-1]) ** 2)))
        self.delta.append(last_layer_error * derivative_value)
        # Back Propagation
        weights_level = -1
        XS_level = -2
        for backcounter in range(self.n_hidden):
            weights = np.delete(self.weights_list[weights_level], 0, axis=0)
            weights = np.transpose(weights)
            WD = np.dot(self.delta[-1], weights)
            derivative_value=[]
            if self.functions=="Sigmoind":
                derivative_value = np.array(self.Xs[XS_level]) * (1 - np.array(self.Xs[XS_level]))
            else:
                derivative_value = np.array((1 - (np.array(self.Xs[XS_level])**2)))
            derivative_value = np.expand_dims(derivative_value, axis=1)
            derivative_value = derivative_value.reshape(1, len(self.Xs[XS_level][0]))
            self.delta.append(WD * derivative_value)
            weights_level = weights_level - 1
            XS_level = XS_level - 1

    def Update_weights(self):
        delta=-1
        Neuron_Values = np.transpose(self.Xs[0])
        Neuron_delta = np.dot(Neuron_Values, self.delta[delta])
        Update_Value = np.array(Neuron_delta) * self.LR
        self.weights_list[0] += (Update_Value)
        delta -= 1
        for w in range(1,len(self.weights_list)):
            Neuron_Value=np.insert(self.Xs[w], 0, self.X0, axis=1)
            Neuron_Value = np.transpose(Neuron_Value)
            Neuron_delta = np.dot(Neuron_Value, self.delta[delta])
            Update_Value = np.array(Neuron_delta)* self.LR
            self.weights_list[w]= (Update_Value) + self.weights_list[w]
            delta-=1
    def test(self):
        print("test")
        exp = [1, 0, -1]
        out={"1":[0,0,0],"0":[0,0,0],"-1":[0,0,0]}
        tru=0
        rownum=0
        for row in self.test_data.values:
            last_layer=self.Get_pred(row[:-1])
            if row[-1] != exp[np.argmax(last_layer[0][0][1:])]:
                print("missclassify at"+str(rownum))
                if row[-1]==1:
                   if exp[np.argmax(last_layer[0][0][1:])]==0:
                       out[str(int(row[-1]))][1]+=1
                   else:
                       out[str(int(row[-1]))][2] += 1
                if row[-1] == 0:
                   if exp[np.argmax(last_layer[0][0][1:])] == 1:
                       out[str(int(row[-1]))][1] += 1
                   else:
                       out[str(int(row[-1]))][2] += 1
                if row[-1] == -1:
                   if exp[np.argmax(last_layer[0][0][1:])] == 1:
                       out[str(int(row[-1]))][1] += 1
                   else:
                       out[str(int(row[-1]))][2] += 1
            else:
                out[str(int(row[-1]))][0] += 1
                tru+=1
            print("Expected:" + str(row[-1])+" Network Output: "+str(exp[np.argmax(last_layer[0][0][1:])]))
            rownum+=1
        self.confusion_matrix(out)
        Totalacc=tru / len(self.test_data.values)*100
        print("Total Accuracy: "+str(Totalacc)+"%")
        return Totalacc

    def Get_pred(self,input):
        last_layer = []
        last_layer.append(1)
        data_input = np.transpose(input)
        data_input = np.expand_dims(data_input, axis=1)
        XS = np.transpose(data_input)
        for w in range(len(self.weights_list)):  # feed forward
            WX = np.dot(XS, self.weights_list[w])
            act_WX = [self.sigmoid(dp) for dp in WX]
            XS = np.insert(act_WX, 0, 1, axis=1)
            last_layer[-1] = XS
        return last_layer

    def confusion_matrix(self,out):
        result = [[out["1"][0], out["1"][1], out["1"][2]],
                  [out["0"][1], out["0"][0], out["0"][2]],
                  [out["-1"][2], out["-1"][1], out["-1"][0]]
                  ]

        df_cm = pd.DataFrame(result, columns=self.classes_list, index=self.classes_list)
        # plt.figure(figsize=(10,7))
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
        plt.show()

    def insert_sample(self,input):
        exp = [1, 0, -1]
        input.insert(0,1)
        last_layer=self.Get_pred(input)
        class_=exp[np.argmax(last_layer[0][0][1:])]
        for classname, class_encode in self.encode_list.items():
            if class_encode == class_:
                print(classname)
                return classname


#L=neuralnetwork("IrisData.txt",["X1","X2","X3","X4"],["Iris-setosa","Iris-virginica","Iris-versicolor"],1,2,1500,0.1,[4,10,5, 3],"Sigmoind")
#L=neuralnetwork("IrisData.txt",["X1","X2","X3","X4"],["Iris-setosa","Iris-virginica","Iris-versicolor"],1,1,250,0.1,[4,12, 3],"Sigmoind")
#L=neuralnetwork("IrisData.txt",["X1","X2","X3","X4"],["Iris-setosa","Iris-virginica","Iris-versicolor"],1,2,5000,0.1,[4,24,12, 3],"Sigmoind")

#L.compile()
#L.insert_sample([5.1,3.5,1.4,0.2])
