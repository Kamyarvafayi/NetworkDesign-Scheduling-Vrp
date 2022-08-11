import numpy as np
import matplotlib.pyplot as plt
import random
# In[]:
class Hub_Loc:
    def __init__(self, Iteration = 1000, Hub_Number = 5, Hub_Distance_Coeff = 0.2):
        self.Iter_Number = Iteration
        self.Hub_Number = Hub_Number
        self.Hub_Distance_Coeff = Hub_Distance_Coeff
    def Find_Distance(self):
        self.Distance = np.array([[np.linalg.norm(self.Input[i]-self.Input[j]) for i in range(self.Input.shape[0])] for j in range(self.Input.shape[0])])
    def Initialization(self):
        self.Initial_Hubs = np.array([random.sample(range(self.Input.shape[0]),self.Hub_Number) for i in range(self.Population_Size)])
        self.Initial_Assignment = np.array([[self.Initial_Hubs[i][np.random.randint(0,self.Hub_Number)] for j in range(self.Input.shape[0])] for i in range(self.Population_Size)])
        for i in range(self.Population_Size):
            for j in range(self.Input.shape[0]):
                if j in self.Initial_Hubs[i]:
                    self.Initial_Assignment[i,j] = j
    def Hub_Mutation(self, Hub_Solution, Hub_Assignment):
        Mutation_Index = np.random.randint(0,self.Hub_Number)
        New_Hub_Solution = Hub_Solution.copy()
        Former_Hub = New_Hub_Solution[Mutation_Index].copy()
        Not_Hub_Indices = np.array([])
        for i in range(self.Input.shape[0]):
            if i not in New_Hub_Solution:
                Not_Hub_Indices = np.append(Not_Hub_Indices, i)
        New_Hub_Solution[Mutation_Index] = Not_Hub_Indices[np.random.randint(0,len(Not_Hub_Indices))]
        New_Hub = New_Hub_Solution[Mutation_Index]
        Hub_Assignment[Hub_Assignment==Former_Hub] = New_Hub
        Hub_Assignment[New_Hub] = New_Hub
        return New_Hub_Solution, Hub_Assignment
    def Hub_Assignment_Swap(self, Hub_Solution, Hub_Assignment):
        Swap_Indices = np.random.randint(0,self.Input.shape[0],size = 2)
        Swap_Indices.sort()
        if (Swap_Indices[0] not in Hub_Solution) and (Swap_Indices[1] not in Hub_Solution):
            temp = Hub_Assignment[Swap_Indices[0]]
            Hub_Assignment[Swap_Indices[0]] = Hub_Assignment[Swap_Indices[1]]
            Hub_Assignment[Swap_Indices[1]] = temp
        return Hub_Assignment
    def Hub_Assignment_reverse(self, Hub_Solution, Hub_Assignment):
        Reverse_Indices = np.random.randint(0,self.Input.shape[0],size = 2)
        Reverse_Indices.sort()
        New_solution = Hub_Assignment.copy()
        flipped_Part = np.flip(New_solution[Reverse_Indices[0]:Reverse_Indices[1]])
        New_solution[Reverse_Indices[0]:Reverse_Indices[1]] = flipped_Part
        for i in range(self.Hub_Number):
            New_solution[Hub_Solution[i]] = Hub_Solution[i]
        return New_solution
    def Hub_Assignment_Change(self, Hub_Solution, Hub_Assignment):
        Index = np.random.randint(0,self.Input.shape[0])
        if Index not in Hub_Solution:
            Hub_Assignment[Index] = Hub_Solution[np.random.randint(0,self.Hub_Number)]
        return Hub_Assignment
    def Find_Objective(self, Hub_Solution, Hub_Assignment):
        Hub_connection_cost = 0
        for i in range(self.Input.shape[0]):
            for j in range(i+1, self.Input.shape[0]):
                Hub_connection_cost += self.Weights[i,j]*self.Distance[i,Hub_Assignment[i]] + self.Weights[i,j]*self.Hub_Distance_Coeff * self.Distance[Hub_Assignment[i], Hub_Assignment[j]] +  self.Weights[i,j]*self.Distance[Hub_Assignment[j],j] 
        Violation_Cost = 0
        for Hub in Hub_Solution:
            Violation_Cost += max(np.sum(self.Weights[Hub_Assignment==Hub])/2 - self.Capacity[Hub], 0)
        return Hub_connection_cost + Violation_Cost*self.Capacity_Violation_Cost
    def Hub_location(self, Input, Weights, Node_Capacity, Capacity_Violation_Cost= 10000, Population_Size = 50):
        self.Input=Input
        self.Weights = Weights
        self.Capacity = Node_Capacity
        self.Capacity_Violation_Cost = Capacity_Violation_Cost
        self.Population_Size = Population_Size
        self.Find_Distance()
        self.Initialization()
        self.All_Hub_Solutions = []
        self.All_Hub_Assignment_Solutions = []
        self.All_Hub_Solutions.append(self.Initial_Hubs)
        self.All_Hub_Assignment_Solutions.append(self.Initial_Assignment)
        
        self.Objectives = np.array([self.Find_Objective(self.Initial_Hubs[i],self.Initial_Assignment[i]) for i in range(self.Population_Size)])
        self.Objectives = self.Objectives.reshape(self.Objectives.shape[0])
        self.Population_Hubs = self.Initial_Hubs[np.argsort(self.Objectives)]
        self.Population_Hub_Assignment = self.Initial_Assignment[np.argsort(self.Objectives)]
        self.Objectives_Sorted = np.sort(self.Objectives)

        for i in range(self.Iter_Number):
            Hub_Mutation_solutions = self.Population_Hubs.copy()
            Assignment_Mutation_solution = self.Population_Hub_Assignment.copy()
            Hub_Swap_solutions = self.Population_Hubs.copy()
            Hub_Assignment_Swap_solutions = self.Population_Hub_Assignment.copy()
            Hub_Reversion_solutions = self.Population_Hubs.copy()
            Hub_Assignment_Reverse_solutions = self.Population_Hub_Assignment.copy()
            Hub_Change_solutions = self.Population_Hubs.copy()
            Hub_Assignment_Change_solutions = self.Population_Hub_Assignment.copy()
            
            # Hub_Mutation
            for k in range(3):
                Hub_Mutation_solutions = self.Population_Hubs.copy()
                Assignment_Mutation_solution = self.Population_Hub_Assignment.copy()
                for j in range(self.Population_Size):
                    New_Solution =  self.Hub_Mutation(Hub_Mutation_solutions[j],Assignment_Mutation_solution[j])
                    self.Population_Hubs = np.append(self.Population_Hubs,np.array([New_Solution[0]]),axis = 0)
                    self.Population_Hub_Assignment =  np.append(self.Population_Hub_Assignment, np.array([New_Solution[1]]), axis=0)
                
            # Hub_Assignment_Swap
            if self.Hub_Number>1:
                for j in range(self.Population_Size):
                    New_Solution =  self.Hub_Assignment_Swap(Hub_Swap_solutions[j],  Hub_Assignment_Change_solutions[j])
                    self.Population_Hubs = np.append(self.Population_Hubs,np.array([Hub_Swap_solutions[j]]),axis = 0)
                    self.Population_Hub_Assignment =  np.append(self.Population_Hub_Assignment, np.array([New_Solution]), axis=0)
                
            # Change
                for k in range(3):
                    for j in range(self.Population_Size):
                        New_Solution =  self.Hub_Assignment_Change(Hub_Change_solutions[j], Hub_Assignment_Swap_solutions[j])
                        self.Population_Hubs = np.append(self.Population_Hubs,np.array([Hub_Change_solutions[j]]),axis = 0)
                        self.Population_Hub_Assignment =  np.append(self.Population_Hub_Assignment, np.array([New_Solution]), axis=0)
                
            # Hub_Assignment_Reversion
                for j in range(self.Population_Size):
                    New_Solution =  self.Hub_Assignment_reverse(Hub_Reversion_solutions[j], Hub_Assignment_Reverse_solutions[j])
                    self.Population_Hubs = np.append(self.Population_Hubs,np.array([Hub_Reversion_solutions[j]]),axis = 0)
                    self.Population_Hub_Assignment =  np.append(self.Population_Hub_Assignment, np.array([New_Solution]), axis=0)
                #print(self.Population_Hubs[-1],"  ", self.Population_Hub_Assignment[-1])
            # removing Duplicates
                for j in range(self.Population_Hubs.shape[0]):
                    self.Population_Hubs[j].sort()
                self.Population = np.append(self.Population_Hubs,self.Population_Hub_Assignment, axis = 1)
                self.Population = np.unique(self.Population, axis=0)
                self.Population_Hubs = np.array(self.Population[:,0:self.Hub_Number],dtype = int)
                self.Population_Hub_Assignment = self.Population[:,self.Hub_Number:]
            # Find Objectives
            self.Objectives = np.array([self.Find_Objective(self.Population_Hubs[k], self.Population_Hub_Assignment[k]) for k in range(self.Population_Hub_Assignment.shape[0])])
            self.Objectives = self.Objectives.reshape(self.Objectives.shape[0])
            self.Population_Hubs = self.Population_Hubs[np.argsort(self.Objectives)]
            self.Population_Hub_Assignment = self.Population_Hub_Assignment[np.argsort(self.Objectives)]
            self.Objectives_Sorted = np.sort(self.Objectives)
            self.Population_Hubs= self.Population_Hubs[0:self.Population_Size,:]
            self.Population_Hub_Assignment = self.Population_Hub_Assignment[0:self.Population_Size,:]
            self.Objectives_Sorted = self.Objectives_Sorted[0:self.Population_Size]
            print("The optimal cost in Iteration {} is: ".format(i+1),self.Objectives_Sorted[0])
# In[]:
np.random.seed(seed = 1)
Sample_Number = 30
#Points_Weights np.random.uniform(0,100,size = [Sample_Number, Sample_Number])
Locations = np.random.uniform(0,100,size = [Sample_Number, 2])
np.random.seed(seed = 1)
Weights = np.random.uniform(1,50,size = [Sample_Number, Sample_Number])
np.random.seed(seed = 1)
Capacity = np.random.uniform(15000,40000,size = [Sample_Number, 1])
Capacity_Violation_Cost = 100000
Weights = Weights + Weights.T
Input = np.array([[np.linalg.norm(Locations[i]-Locations[j]) for i in range(Sample_Number)] for j in range(Sample_Number)])
Hub_Number = 5

Hub_Object = Hub_Loc(Iteration = 300, Hub_Number = Hub_Number , Hub_Distance_Coeff = 0.2)
Hub = Hub_Object.Hub_location(Locations, Weights, Node_Capacity = Capacity, Capacity_Violation_Cost = Capacity_Violation_Cost, Population_Size = 50)
Hubs = Hub_Object.Population_Hubs[0]
Assignments = Hub_Object.Population_Hub_Assignment[0]
Dist_Weights = Hub_Object.Weights 
print(Hubs)
print(Assignments) 
# In[]:
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(Locations[:,0],Locations[:,1], marker = 'o', c = 'red')
for hub in range(Hub_Number):               
    plt.scatter(Locations[Hubs[hub], 0],Locations[Hubs[hub], 1], linewidth = 10, marker = '>', c = 'green')
for Points in range(Sample_Number):
    plt.text(Locations[Points,0]+0.02, Locations[Points,1]+0.02, '{}'.format(Points)) 
for hub in range(Hub_Number):
    for Points in range(Sample_Number):
        if Assignments[Points] == Hubs[hub]:
            plt.plot(np.array([Locations[Points,0], Locations[Hubs[hub],0]]),np.array([Locations[Points,1], Locations[Hubs[hub],1]]), color = 'black' )
