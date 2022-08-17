import matplotlib.pyplot as plt
import numpy as np
import random
# In[]: Parallel Machines Scheduling
class parallel_Machine_Scheduling_Optimized_Sa:
    def __init__(self, Iteration = 2000, Temp = 100, Temp_Coeff = 0.95):
        self.Iter_Num = Iteration
        self.Temp = Temp
        self.Temp_Coeff = Temp_Coeff
    def Initialization(self):
        self.First_solution = [[] for i in range(self.Machine_Num)]
        Work_Order = np.arange(0,self.Work_Num, dtype = int)
        random.shuffle(Work_Order)
        for works in Work_Order:
            rand = np.random.rand()
            for machine in range(self.Machine_Num):
                if rand < (machine+1)/self.Machine_Num and rand > max( (machine)/self.Machine_Num, 0) :
                    self.First_solution[machine].append(works)
        
        for machine in range(self.Machine_Num):
            self.First_solution[machine] = np.array(self.First_solution[machine]) 
        print(self.First_solution)
    def Insert(self, Sol):
        Solution = Sol.copy()
        Insert_Machine_Indices = random.sample(range(self.Machine_Num), 2)
        if len(Solution[Insert_Machine_Indices[0]])>0 and len(Solution[Insert_Machine_Indices[1]])>0:
            Insert_Index1 = np.random.randint(0, len(Solution[Insert_Machine_Indices[0]]))
            Insert_Index2 = np.random.randint(0, len(Solution[Insert_Machine_Indices[1]]))
            Rand = np.random.rand()
            if Rand>0.1:
                if  len(Solution[Insert_Machine_Indices[1]])>1:    
                    Solution[Insert_Machine_Indices[0]] = np.insert(Solution[Insert_Machine_Indices[0]], Insert_Index1, Solution[Insert_Machine_Indices[1]][Insert_Index2])
                    Solution[Insert_Machine_Indices[1]] = np.append(Solution[Insert_Machine_Indices[1]][:Insert_Index2], Solution[Insert_Machine_Indices[1]][Insert_Index2+1:])
            else:
                if  len(Solution[Insert_Machine_Indices[1]])>1:    
                    Solution[Insert_Machine_Indices[0]] = np.append(Solution[Insert_Machine_Indices[0]], Solution[Insert_Machine_Indices[1]][Insert_Index2])
                    Solution[Insert_Machine_Indices[1]] = np.append(Solution[Insert_Machine_Indices[1]][:Insert_Index2], Solution[Insert_Machine_Indices[1]][Insert_Index2+1:])
                 
        return Solution
    def Find_Objective_Function(self, Sol):
        Solution = Sol.copy()
        Machines_Work = []
        Due_Time_Violation = 0
        for i in range(self.Machine_Num):
            Cumulation_Work = 0
            for j in range(len(Solution[i])):
                if j == len(Solution[i])-1:
                    Cumulation_Work += self.Work_Duration[int(Solution[i][j])]
                    Due_Time_Violation += max(0, Cumulation_Work - self.Due_time[int(Solution[i][j])])
                else:
                    Cumulation_Work += self.Work_Duration[int(Solution[i][j])]
                    Due_Time_Violation += max(0, Cumulation_Work - self.Due_time[int(Solution[i][j])])
                    Cumulation_Work += self.Warm_Up
            Machines_Work.append(Cumulation_Work)
        Obj_Func = max(Machines_Work) + self.Due_time_Cost*Due_Time_Violation
        return Obj_Func
    def Fit_Sa_To_Parallel_Machine_Scheduling(self, Work_Duration, Due_time, Due_Violation_Cost = 10000, Machine_Num = 3, Warm_Up = 5):
        self.Work_Duration = Work_Duration
        self.Work_Num = self.Work_Duration.shape[0]
        self.Machine_Num = Machine_Num
        self.Due_time = Due_time
        self.Due_time_Cost = Due_Violation_Cost
        self.Warm_Up = Warm_Up
        self.Initialization()
        self.Best_Obj = self.Objectives = np.array([self.Find_Objective_Function(self.First_solution)])
        Best_Sol = self.First_solution
        while self.Temp>=1:
            for iteration in range(self.Iter_Num):
                Best_Sol_Copy = Best_Sol.copy()
                candid_Solution = self.Insert(Best_Sol_Copy)
                if self.Find_Objective_Function(candid_Solution) <= self.Find_Objective_Function(Best_Sol_Copy):
                    Best_Sol  = candid_Solution.copy()
                    self.Best_Obj = self.Find_Objective_Function(candid_Solution)
                    #self.Best_Obj = self.Find_Objective_Function(self.Best_Sol)
                    self.Objectives = np.append(self.Objectives, np.array([self.Best_Obj]))
                else:
                    DeltaObjective = self.Find_Objective_Function(Best_Sol_Copy)- self.Find_Objective_Function(candid_Solution)
                    rand2 = np.random.uniform(0,1)
                    if rand2 < np.exp(DeltaObjective*10/self.Temp) and DeltaObjective>-20 :    
                         Best_Sol = candid_Solution.copy()
                         self.Best_Obj = self.Find_Objective_Function(candid_Solution)
                         self.Objectives = np.append(self.Objectives, np.array([self.Best_Obj]))
                print("The optimal cost for Temp {} and in Iteration {} is: ".format(self.Temp,iteration+1),self.Best_Obj)
            self.Temp *= self.Temp_Coeff
        self.Best_Sol = Best_Sol
# In[]:
np.random.seed(seed=1)
Work_Duration = np.random.randint(1,100,size = 300)
Machine_Number = 40
Due_time = 600*np.ones([Work_Duration.shape[0]])
Due_time[0] = 40
Due_time[1] = 80
Due_time[10] = 80
Due_time[15] = 90
Due_time[12] = 80
Warm_up = 15
PMObject = parallel_Machine_Scheduling_Optimized_Sa(Iteration = 1200, Temp = 1000, Temp_Coeff = 0.8 )
PMObject.Fit_Sa_To_Parallel_Machine_Scheduling(Work_Duration, Due_time, Machine_Num = Machine_Number , Warm_Up = Warm_up)
Best_Sol = PMObject.Best_Sol
print(PMObject.Find_Objective_Function(PMObject.Best_Sol))             
Objectives = PMObject.Objectives 
# In[]: chart
plt.figure()
width = 0.6
gap = 0.2
color = ['red','blue',[0.9,0.5,0.6],'green','cyan','purple','grey','brown','pink', 'orange',[0.1,0.6,0.9],[1,0.5,0.3], [0.5,1,0.3],[0.3,0.5,1],[0.5,0.2,0.6]]
for i in range(Machine_Number):
    Cum_Duration = 0
    np.random.seed(seed=np.random.randint(0,10))
    y = i+1
    for j in range(len(Best_Sol[i])):
        plt.fill([Cum_Duration,Cum_Duration,Cum_Duration+Work_Duration[int(Best_Sol[i][j])],Cum_Duration+Work_Duration[int(Best_Sol[i][j])]], [y,y+width,y+width, y], color=[np.random.randint(0,256)/255,np.random.randint(0,256)/255, np.random.randint(0,256)/255 ])
        plt.text(Cum_Duration+Work_Duration[int(Best_Sol[i][j])]/2 -3, y+width/2, int(Best_Sol[i][j]))
        Cum_Duration += Work_Duration[int(Best_Sol[i][j])]+ Warm_up
plt.axvline(x=Objectives[-1]%10000, ymin=0, ymax=Machine_Number,linestyle='dotted', color = "black")
plt.text(Objectives[-1]%10000+2, Machine_Number/2, "T= {}".format(Objectives[-1]%10000))
plt.xlim([0,Objectives[-1]%10000+20])
plt.ylim([1,Machine_Number+1])