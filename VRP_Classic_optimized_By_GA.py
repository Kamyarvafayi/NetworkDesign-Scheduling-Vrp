import numpy as np
import random as random
# In[]:
class Classic_VRP:
    def __init__(self, Iteration = 100, Vehicle_Num = 3, Population_Size = 5):
        self.Iter_Number = Iteration
        self.Vehicle_Num = Vehicle_Num
        self.Population_Size = Population_Size
    def Initialization(self):
        self.City_Orders_Initial_Solution = np.array([[np.random.rand() for i in range(self.Input.shape[0])] for j in range(self.Population_Size)])
        self.Vehicle_Initial_Solution = np.array([[np.random.randint(1,self.Vehicle_Num + 1) for i in range(self.Input.shape[0])] for j in range(self.Population_Size)])
    def Vehicle_CrossOver(self, Vehicle1, Vehicle2):
        Random_Coeff = np.array([np.random.randint(0,2,size = self.Input.shape[0])])
        return Vehicle1*Random_Coeff + Vehicle2*(1-Random_Coeff), Vehicle2*Random_Coeff + Vehicle1*(1-Random_Coeff)
    def Vehicle_Mutation(self, Vehicle):
        Mutation_Size = int(max(np.ceil(self.Input.shape[0]/5), 3))
        Mutation_Indices = random.sample(range(self.Input.shape[0]), Mutation_Size)
        New_Vehicle = Vehicle.copy()
        for Vehicle_Index in Mutation_Indices:
            Random_Vehicle = np.random.rand()
            New_Vehicle[Vehicle_Index] = min(np.floor(Random_Vehicle*self.Vehicle_Num), self.Vehicle_Num-1)+1
        return New_Vehicle
    def City_Order_Crossover(self, City_Order1, City_Order2):
        Random_Coeff = np.array([np.random.uniform(0,1,size = self.Input.shape[0])])
        Random_Indices = random.sample(range(0,self.Input.shape[0]), 2)
        Random_Indices.sort()
        City_Order1_Copy = City_Order1.copy()
        City_Order2_Copy = City_Order2.copy()
        temp = City_Order1_Copy[Random_Indices[0]:Random_Indices[1]]
        City_Order1_Copy[Random_Indices[0]:Random_Indices[1]] = City_Order2_Copy[Random_Indices[0]:Random_Indices[1]]
        City_Order2_Copy[Random_Indices[0]:Random_Indices[1]] = temp
        return City_Order1_Copy , City_Order2_Copy
    def City_Order_Mutation(self, City_Order):
        Mutation_Speed = 0.2
        Mutation_Size = int(max(np.ceil(self.Input.shape[0]/5), 3))
        Mutation_Indices = random.sample(range(self.Input.shape[0]), Mutation_Size)
        New_Order = City_Order.copy()
        for City_Index in Mutation_Indices:
            New_Order[City_Index] = New_Order[City_Index] + Mutation_Speed*np.random.randn()
            if New_Order[City_Index]>1:
                New_Order[City_Index] = 1
            elif New_Order[City_Index]< 0:
                New_Order[City_Index] = 0
        return New_Order  
    def Find_Objective(self, Vehicle_Order, City_Order):
        Vehicle_Sorted = Vehicle_Order[np.argsort(City_Order)]
        City_Sorted = np.arange(1,self.Input.shape[0]+1,1)
        City_Sorted = City_Sorted[np.argsort(City_Order)]
        
        Cost_Of_Vehicle = 0
        for City in range(1, self.Input.shape[0]+1):
            Cost_Of_Vehicle += self.Vehicle_Cost[Vehicle_Sorted[City_Sorted==City]-1,City-1]
        
        Transportation_Cost = 0
        for Vehicle in range(1, self.Vehicle_Num+1):
            First_City = True
            Find_Vehicle = False
            for City in City_Sorted: 
                if Vehicle_Order[City-1]==Vehicle:
                    if First_City:
                        Transportation_Cost += self.Input[City-1,City-1]
                        First_City = False
                        Origin = City-1
                    else:
                        Transportation_Cost += self.Input[Origin,City-1]
                        Origin = City-1
                    Find_Vehicle = True
            if Find_Vehicle:
                Transportation_Cost += self.Input[Origin,Origin]
        
        return Cost_Of_Vehicle + Transportation_Cost

    def Fit_Classic_Vrp(self, Input, Vehicle_Cost):
        self.Input = Input
        self.Vehicle_Cost = Vehicle_Cost
        self.All_Vehicle_Solutions = []
        self.All_City_Order_Solutions = []
        self.Initialization()
        self.All_Vehicle_Solutions.append(self.Vehicle_Initial_Solution)
        self.All_City_Order_Solutions.append(self.City_Orders_Initial_Solution)
        self.Objectives = np.array([self.Find_Objective(self.Vehicle_Initial_Solution[i],self.City_Orders_Initial_Solution[i]) for i in range(self.Population_Size)])
        self.Objectives = self.Objectives.reshape(self.Objectives.shape[0])
        self.Population_Vehicle = self.Vehicle_Initial_Solution[np.argsort(self.Objectives)]
        self.Population_City = self.City_Orders_Initial_Solution[np.argsort(self.Objectives)]
        self.Objectives_Sorted = np.sort(self.Objectives)
        for i in range(self.Iter_Number):
            Vehicle_Cross_Over_solutions = self.Population_Vehicle.copy()
            Cities_Cross_Over_solutions = self.Population_City.copy()
            Vehicle_Mutation_solutions = self.Population_Vehicle.copy()
            Cities_Mutation_solutions = self.Population_City.copy()
        # Crossover
            Cross_Over_Indices = random.sample(range(0,self.Population_Size),30)
            Cross_Over_Indices2 = Cross_Over_Indices.copy()
            for j in Cross_Over_Indices:
                Cross_Over_Indices2 = np.delete(Cross_Over_Indices2, np.where(Cross_Over_Indices2 == j))
                for k in Cross_Over_Indices2:
                    self.Population_Vehicle =  np.append(self.Population_Vehicle, self.Vehicle_CrossOver(Vehicle_Cross_Over_solutions[j],Vehicle_Cross_Over_solutions[k])[0], axis=0)
                    self.Population_Vehicle =  np.append(self.Population_Vehicle, self.Vehicle_CrossOver(Vehicle_Cross_Over_solutions[j],Vehicle_Cross_Over_solutions[k])[1], axis=0)
                    self.Population_City =  np.append(self.Population_City, np.array([Cities_Cross_Over_solutions[j]]), axis=0)
                    self.Population_City =  np.append(self.Population_City, np.array([Cities_Cross_Over_solutions[k]]), axis=0)
                    self.Population_City =  np.append(self.Population_City, np.array([self.City_Order_Crossover(Cities_Cross_Over_solutions[j],Cities_Cross_Over_solutions[k])[0]]), axis=0)
                    self.Population_City =  np.append(self.Population_City, np.array([self.City_Order_Crossover(Cities_Cross_Over_solutions[j],Cities_Cross_Over_solutions[k])[1]]), axis=0)
                    self.Population_Vehicle =  np.append(self.Population_Vehicle, np.array([Vehicle_Cross_Over_solutions[j]]), axis=0)
                    self.Population_Vehicle =  np.append(self.Population_Vehicle, np.array([Vehicle_Cross_Over_solutions[k]]), axis=0)
        # Mutation
            Mutation_Indices = random.sample(range(0,self.Population_Size),int(self.Population_Size*0.8))
            for k in range(1):
                for j in Mutation_Indices:
                    self.Population_Vehicle =  np.append(self.Population_Vehicle, np.array([self.Vehicle_Mutation(Vehicle_Mutation_solutions[j])]), axis=0)
                    self.Population_City =  np.append(self.Population_City, np.array([Cities_Cross_Over_solutions[j]]), axis=0)
                    self.Population_City =  np.append(self.Population_City, np.array([self.City_Order_Mutation(Cities_Mutation_solutions[j])]), axis=0)
                    self.Population_Vehicle =  np.append(self.Population_Vehicle, np.array([Vehicle_Cross_Over_solutions[j]]), axis=0)     
        # Removing Duplicates from Population 
            self.Population = np.append(self.Population_Vehicle,self.Population_City, axis = 1)
            self.Population = np.unique(self.Population, axis=0)
            self.Population_Vehicle = np.array(self.Population[:,0:self.Input.shape[0]],dtype = int)
            self.Population_City = self.Population[:,self.Input.shape[0]:]
        # Find Objective
            self.Objectives = np.array([self.Find_Objective(self.Population_Vehicle[i], self.Population_City[i]) for i in range(self.Population_City.shape[0])])
            self.Objectives = self.Objectives.reshape(self.Objectives.shape[0])
            self.Population_City = self.Population_City[np.argsort(self.Objectives)]
            self.Population_Vehicle = self.Population_Vehicle[np.argsort(self.Objectives)]
            self.Objectives_Sorted = np.sort(self.Objectives)
            self.Population_City = self.Population_City[0:self.Population_Size,:]
            self.Population_Vehicle = self.Population_Vehicle[0:self.Population_Size,:]
            self.Objectives_Sorted = self.Objectives_Sorted[0:self.Population_Size]
            print("The optimal cost in Iteration {} is: ".format(i+1),self.Objectives_Sorted[0])
# In[]:
Input = np.array([[100,150,200,50,0],[150,200,250,220,500],[200,250,300,300,600],[50,220,300,400,1000],[0,500,600,1000,100]])
#Input = np.array([[0,150,200,50],[150,0,250,220],[200,250,0,300],[50,220,300,0]])
Vehicle_Cost = np.array([[100,200,300],[70,50,0],[300,500,200]])
Vehicle_Cost = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0]])
Vehicle_Cost = np.array([[100,350,650,200,100],[250,150,500,150,50],[300,250,230,220,400]])
VRP_Object = Classic_VRP(Iteration = 50,Population_Size = 500, Vehicle_Num=3)
Vrp = VRP_Object.Fit_Classic_Vrp(Input/100, Vehicle_Cost/100)
Vrp_City = VRP_Object.Population_City
Vrp_Vehicle = VRP_Object.Population_Vehicle
Vehicle_Sorted = Vrp_Vehicle[0][np.argsort(Vrp_City[0])]
City_Sorted = np.arange(1,Input.shape[0]+1,1)
City_Sorted = City_Sorted[np.argsort(Vrp_City[0])]  
print(Vehicle_Sorted)
print(City_Sorted)   
# In[]:
np.random.seed(seed = 1)
Input2 = np.random.randint(100,500,size = [200,200])
Vehicle_Number = 5
np.random.seed(seed = 1) 
Vehicle_Cost2 = np.random.randint(10,100,size = [Vehicle_Number,200])
Vehicle_Cost2[0] = 500000
VRP_Object2 = Classic_VRP(Iteration = 100,Population_Size = 500, Vehicle_Num=Vehicle_Number)
Vrp2 = VRP_Object2.Fit_Classic_Vrp(Input2, Vehicle_Cost2)
Vrp_City2 = VRP_Object2.Population_City
Vrp_Vehicle2 = VRP_Object2.Population_Vehicle
Vehicle_Sorted2 = Vrp_Vehicle2[0][np.argsort(Vrp_City2[0])]
City_Sorted2 = np.arange(1,Input2.shape[0]+1,1)
City_Sorted2 = City_Sorted2[np.argsort(Vrp_City2[0])]  
print(Vehicle_Sorted2)
print(City_Sorted2) 

# In[]:
np.random.seed(seed = 1)
Sample_Number = 100
Locations = np.random.uniform(0,100,size = [Sample_Number, 2])
Input3 = np.array([[np.linalg.norm(Locations[i]-Locations[j]) for i in range(Sample_Number)] for j in range(Sample_Number)])
Depot_Position = np.average(Locations, axis = 0)
Vehicle_Number2 = 12
np.random.seed(seed = 1)
Vehicle_Cost3 = np.random.randint(0,1,size = [Vehicle_Number2,Sample_Number])
for i in range(Sample_Number):
    Input3[i,i] = np.linalg.norm(Depot_Position-Locations[i]) 

VRP_Object3 = Classic_VRP(Iteration = 200,Population_Size = 500, Vehicle_Num=Vehicle_Number2)
Vrp3 = VRP_Object3.Fit_Classic_Vrp(Input3, Vehicle_Cost3)
Vrp_City3 = VRP_Object3.Population_City
Vrp_Vehicle3 = VRP_Object3.Population_Vehicle
Vehicle_Sorted3 = Vrp_Vehicle3[0][np.argsort(Vrp_City3[0])]
City_Sorted3 = np.arange(1,Input3.shape[0]+1,1)
City_Sorted3 = City_Sorted3[np.argsort(Vrp_City3[0])]  
print(Vehicle_Sorted3)
print(City_Sorted3)
Vehicles = Vehicle_Sorted3[np.argsort(City_Sorted3)]

import matplotlib.pyplot as plt
plt.figure()
plt.scatter(Locations[:,0],Locations[:,1], marker = 'o', c = 'black')                
plt.scatter(Depot_Position[0],Depot_Position[1], marker = 'x', c = 'red')
# In[]: Charts
color = ['red','blue','black','green','cyan','purple','grey','brown','pink', 'orange',[1,0.5,0.3],[0.5,1,0.3],[0.3,0.5,1],[0.5,0.2,0.6]]
marker =['x','o','s','<','>','^']
plt.figure()
for i in range(1,Vehicle_Number2+1):
    plt.scatter(Locations[Vehicles==i,0],Locations[Vehicles==i,1], marker = '>',c = color[i-1])
for Points in range(Sample_Number):
    plt.text(Locations[Points,0]+0.02, Locations[Points,1]+0.02, '{}'.format(Points+1))
plt.scatter(Depot_Position[0],Depot_Position[1], marker = 's', linewidths = 7, c = [1,0.5,0.3])
plt.text(Depot_Position[0]+0.02,Depot_Position[1]+0.02, "Depot")
for i in range(1,Vehicle_Number2+1):
    Locations_Copy = Locations.copy()
    Locations_Copy = np.insert(Locations_Copy,0,Depot_Position,axis = 0)
    Locations_Copy = np.append(Locations_Copy,np.array([Depot_Position]), axis=0)
    City_Sorted3_Copy = City_Sorted3.copy()
    City_Sorted3_Copy = np.insert(City_Sorted3_Copy,0,np.array([0]),axis = 0)
    City_Sorted3_Copy = np.append(City_Sorted3_Copy,np.array([0]),axis = 0)
    Vehicle_Sorted3_Copy = Vehicle_Sorted3.copy()
    Vehicle_Sorted3_Copy = np.insert(Vehicle_Sorted3_Copy,0,np.array([i]),axis = 0)
    Vehicle_Sorted3_Copy = np.append(Vehicle_Sorted3_Copy,np.array([i]),axis = 0)
    #plt.plot(Locations[origin,:],Locations[City_Sorted3[j]-1,:], color = color[i])
    #plt.plot(Locations[City_Sorted3[Vehicle_Sorted3==i]-1,0],Locations[City_Sorted3[Vehicle_Sorted3==i]-1,1], color = color[i])
    plt.plot(Locations_Copy[City_Sorted3_Copy[Vehicle_Sorted3_Copy==i],0],Locations_Copy[City_Sorted3_Copy[Vehicle_Sorted3_Copy==i],1],'-', color = color[i-1])
    plt.plot(Locations_Copy[0,0],Locations_Copy[0,1], color = color[i-1])
    plt.show()