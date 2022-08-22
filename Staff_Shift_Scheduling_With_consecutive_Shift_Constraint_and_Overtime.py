import numpy as np
import matplotlib.pyplot as plt
# In[]:
class Staff_Shift_Scheduling:
    def __init__(self, Iter_Num = 10, Population_Size = 50):
        self.Max_Iter = Iter_Num
        self.Population_Size = Population_Size
    def Initialization (self):
        self.Initial_Solution = np.array([[[np.random.rand() for i in range(self.Staff_Num)] for j in range(self.Shift_Num)] for k in range(self.Population_Size)])
    def Cross_Over_Columns(self, Solution1, Solution2):
        New_Solution1 = Solution1.copy()
        New_Solution2 = Solution2.copy()
        Cross_Column_Index = np.random.randint(0,self.Staff_Num)
        Cross_Row_Index = np.random.randint(1,self.Shift_Num)
        temp = New_Solution2[Cross_Row_Index-1:,Cross_Column_Index]
        New_Solution2 [Cross_Row_Index-1:,Cross_Column_Index] = New_Solution1[Cross_Row_Index-1:,Cross_Column_Index]
        New_Solution1[Cross_Row_Index-1:,Cross_Column_Index] = temp
        return New_Solution1, New_Solution2
    def Cross_Over_Rows(self, Solution1, Solution2):
        New_Solution1 = Solution1.copy()
        New_Solution2 = Solution2.copy()
        Cross_Column_Index = np.random.randint(1,self.Staff_Num)
        Cross_Row_Index1 = np.random.randint(0,self.Shift_Num)
        if Cross_Row_Index1%2==1:
            Cross_Row_Index2 = Cross_Row_Index1-1
            temp = New_Solution2[Cross_Row_Index2:Cross_Row_Index1,Cross_Column_Index-1:]
            New_Solution2 [Cross_Row_Index2:Cross_Row_Index1,Cross_Column_Index-1:] = New_Solution1[Cross_Row_Index2:Cross_Row_Index1,Cross_Column_Index-1:]
            New_Solution1[Cross_Row_Index2:Cross_Row_Index1,Cross_Column_Index-1:] = temp
        else:
            Cross_Row_Index2 = Cross_Row_Index1+1
            temp = New_Solution2[Cross_Row_Index1:Cross_Row_Index2,Cross_Column_Index-1:]
            New_Solution2 [Cross_Row_Index1:Cross_Row_Index2,Cross_Column_Index-1:] = New_Solution1[Cross_Row_Index1:Cross_Row_Index2,Cross_Column_Index-1:]
            New_Solution1[Cross_Row_Index1:Cross_Row_Index2,Cross_Column_Index-1:] = temp
        return New_Solution1, New_Solution2
    def Row_Mutation(self, Solution):
        New_Solution = Solution.copy()
        Mutation_Row_Index = np.random.randint(0,self.Shift_Num)
        Column_Index = np.random.randint(1,self.Staff_Num)
        Mutation_step = np.array([np.random.randn() for i in range(self.Staff_Num)])
        New_Solution[Mutation_Row_Index,Column_Index-1:] = Mutation_step[Column_Index-1:]*0.1 + New_Solution[Mutation_Row_Index,Column_Index-1:]
        New_Solution[Mutation_Row_Index, New_Solution[Mutation_Row_Index,:]>1] = 1
        New_Solution[Mutation_Row_Index, New_Solution[Mutation_Row_Index,:]<0] = 0
        return New_Solution
    def Column_Mutation(self, Solution):
        New_Solution = Solution.copy()
        Mutation_Column_Index = np.random.randint(0,self.Staff_Num)
        Row_Index = np.random.randint(1,self.Shift_Num)
        Mutation_step = np.array([np.random.randn() for i in range(self.Shift_Num)])
        New_Solution[Row_Index-1:,Mutation_Column_Index] = Mutation_step[Row_Index-1:]*0.1 + New_Solution[Row_Index-1:,Mutation_Column_Index]
        New_Solution[New_Solution[:,Mutation_Column_Index]>1,Mutation_Column_Index] = 1
        New_Solution[New_Solution[:,Mutation_Column_Index]<0,Mutation_Column_Index] = 0
        return New_Solution
    def Mutation(self, Solution):
        New_Solution = Solution.copy()
        Column_Index = np.random.randint(0,self.Staff_Num)
        Row_Index = np.random.randint(0,self.Shift_Num)
        if New_Solution[Row_Index,Column_Index]<0.5:
            New_Solution[Row_Index,Column_Index] = 1
        else:
            New_Solution[Row_Index,Column_Index] = 0
        return New_Solution
    def One_Zero_Mutation(self, Solution):
        New_Solution = Solution.copy()
        Column_Index = np.random.randint(0,self.Staff_Num)
        Row_Index = np.random.randint(0,self.Shift_Num-1)
        if New_Solution[Row_Index,Column_Index]<0.5:
            New_Solution[Row_Index,Column_Index] = 1
            New_Solution[Row_Index+1,Column_Index] = 0
        else:
            New_Solution[Row_Index,Column_Index] = 0
            New_Solution[Row_Index+1,Column_Index] = 1
        return New_Solution
    def Find_Objective(self, Solution):
        Sol = np.zeros([self.Shift_Num, self.Staff_Num])
        for staff in range(self.Staff_Num):
            for shift in range(1,self.Shift_Num):
                if shift%2 == 1:
                    if Solution[shift, staff]>=0.5 or Solution[shift-1, staff]>=0.5:
                        if Solution[shift, staff] > Solution[shift-1, staff]:
                            Sol[shift, staff] = 1
                        else:
                            Sol[shift-1, staff] = 1
        #OverWork = np.count_nonzero(np.where(np.sum(Sol, axis = 0)>=self.Shift_Num/2, 1, 0))
        OverWork = np.array([max(np.sum(Sol[:,i])-self.Shift_Num/2+1, 0) for i in range(self.Staff_Num)])
        Working_Two_Consecutive_Shifts_Count = 0
        for i in range(self.Shift_Num-1):
            for j in range(self.Staff_Num):
                if Sol[i,j]==1 and Sol[i+1,j]==1:
                    Working_Two_Consecutive_Shifts_Count += 1
        Demand_Violation = np.array([max(0,self.Demand[shift]-np.sum(Sol[shift, :])) for shift in range(self.Shift_Num)]) 
        return self.Wage*5*np.sum(OverWork)+ self.Wage*np.sum(Sol) + np.sum(Demand_Violation)*self.Demand_Violation_Cost + 10000*self.Wage*Working_Two_Consecutive_Shifts_Count
    
    def Translate_Solution(self, Solution):
        Sol = np.zeros([self.Shift_Num, self.Staff_Num])
        for staff in range(self.Staff_Num):
            for shift in range(1,self.Shift_Num):
                if shift%2 == 1:
                    if Solution[shift, staff]>=0.5 or Solution[shift-1, staff]>=0.5:
                        if Solution[shift, staff] > Solution[shift-1, staff]:
                            Sol[shift, staff] = 1
                        else:
                            Sol[shift-1, staff] = 1
        return Sol         
    def Fit_GA_to_Staff_Work_Scheduling(self, Demand, Staff_Num = 60, Shift_Num = 14, Demand_Violation_Cost = 10000, Staff_Wage = 100, Cross_Over_ratio = 0.1):
        self.Demand = Demand
        self.Staff_Num = Staff_Num
        self.Shift_Num = Shift_Num
        self.Demand_Violation_Cost = Demand_Violation_Cost
        self.Wage = Staff_Wage
        self.Cross_Over_ratio = Cross_Over_ratio
        self.Initialization()
        self.Objectives = np.array([self.Find_Objective(self.Initial_Solution[i,:,:]) for i in range(self.Population_Size)])
        self.Population = self.Initial_Solution[np.argsort(self.Objectives),:,:]
        self.Objectives_Sorted = np.sort(self.Objectives)
        for iteration in range(self.Max_Iter):
            Column_CrossOVer_Solution = self.Population.copy()
            Row_CrossOver_Solution = self.Population.copy()
            Row_Mutation_Solution = self.Population.copy()
            Column_Mutation_Solution = self.Population.copy()
            Mutation_Solution = self.Population.copy()
            One_Zero_Mutation_Solution = self.Population.copy()
        # Cross_Over
            Cross_Over_Rand = np.random.rand()
            if Cross_Over_Rand>0.5:
                for i in range(self.Population_Size):
                    if np.random.rand()>1-self.Cross_Over_ratio:
                        for j in range(i+1, self.Population_Size):
                            self.Population = np.append(self.Population, self.Cross_Over_Columns(Column_CrossOVer_Solution[i,:,:], Column_CrossOVer_Solution[j,:,:])[0].reshape([1,self.Shift_Num, self.Staff_Num]), axis = 0)
                            self.Population = np.append(self.Population, self.Cross_Over_Columns(Column_CrossOVer_Solution[i,:,:], Column_CrossOVer_Solution[j,:,:])[1].reshape([1,self.Shift_Num, self.Staff_Num]), axis = 0)
            else:
                for i in range(self.Population_Size):
                    if np.random.rand()>self.Cross_Over_ratio:
                        for j in range(i+1, self.Population_Size):
                            self.Population = np.append(self.Population, self.Cross_Over_Rows(Row_CrossOver_Solution[i,:,:], Row_CrossOver_Solution[j,:,:])[0].reshape([1,self.Shift_Num, self.Staff_Num]), axis = 0)
                            self.Population = np.append(self.Population, self.Cross_Over_Rows(Row_CrossOver_Solution[i,:,:], Row_CrossOver_Solution[j,:,:])[1].reshape([1,self.Shift_Num, self.Staff_Num]), axis = 0)
        # Mutation
            for j in range(20):
                Mutation_Solution = self.Population[0:self.Population_Size]
                for i in range(self.Population_Size):
                    self.Population = np.append(self.Population, self.Mutation(Mutation_Solution[i,:,:]).reshape([1,self.Shift_Num, self.Staff_Num]), axis = 0)    
        # Make one or zero Mutation
            for j in range(20):
                Mutation_Solution = self.Population[0:self.Population_Size]
                for i in range(self.Population_Size):
                    self.Population = np.append(self.Population, self.One_Zero_Mutation(One_Zero_Mutation_Solution[i,:,:]).reshape([1,self.Shift_Num, self.Staff_Num]), axis = 0)
        # Row or Column Mutation   
            Mutation_Rand = np.random.rand()
            if Mutation_Rand>0:
                for i in range(self.Population_Size):
                    if np.random.rand()>0.1:
                        self.Population = np.append(self.Population, self.Row_Mutation(Row_Mutation_Solution[i,:,:]).reshape([1,self.Shift_Num, self.Staff_Num]), axis = 0) 
            else:
                for i in range(self.Population_Size):
                    if np.random.rand()>0.1:
                        self.Population = np.append(self.Population, self.Column_Mutation(Column_Mutation_Solution[i,:,:]).reshape([1,self.Shift_Num, self.Staff_Num]), axis = 0) 
        # Creating New Population
            self.Objectives = np.array([self.Find_Objective(self.Population[i,:,:]) for i in range(self.Population.shape[0])])
            self.Population = self.Population[np.argsort(self.Objectives),:,:]
            self.Objectives_Sorted = np.sort(self.Objectives)
            self.Population = self.Population[0:self.Population_Size, :, :]
            self.Objectives_Sorted = self.Objectives_Sorted[0:self.Population_Size] 
            print("Objective in Iteration {} is {}.".format(iteration+1, self.Objectives_Sorted[0]))       
# In[]:
Staff_Scheduling = Staff_Shift_Scheduling(Iter_Num= 1000, Population_Size=5)
Shift_Number = 14
Staff_Number = 60
staff_wage = 100
Demand_Violation_Cost = 1000000
Cross_Ratio = 0.1
np.random.seed(seed=1)
Demand = np.array([np.random.randint(20, 31) for i in range(Shift_Number)])
Staff_Scheduling.Fit_GA_to_Staff_Work_Scheduling(Demand = Demand, Staff_Num = Staff_Number, Shift_Num = Shift_Number, Demand_Violation_Cost = Demand_Violation_Cost, Staff_Wage = staff_wage, Cross_Over_ratio  = Cross_Ratio)
Initial_Sol = Staff_Scheduling.Initial_Solution
Objectives = Staff_Scheduling.Objectives_Sorted
Population = Staff_Scheduling.Population
Final_Solution = Staff_Scheduling.Translate_Solution(Population[0])
print(np.sum(Final_Solution, axis = 1) - Demand)
# In[]: chart
plt.figure()
width = 1
length = 1
color = ['red',[0.25,0.7,0.3],[0.9,0.5,0.6],'green','cyan','purple','grey','brown','pink', 'orange',[0.1,0.6,0.9],[1,0.5,0.3], [0.5,1,0.3],[0.3,0.5,1],[0.5,0.2,0.6]]
for i in range(1, Shift_Number+1):
    for j in range(1, Staff_Number+1):
        if Final_Solution[i-1, j-1]==1:
            plt.fill([j-0.5 , j-0.5, j-0.5+length,j-0.5+length], [i-0.5,i-0.5+width,i-0.5+width, i-0.5], color=color[(j-1)%14])
            plt.text(j-0.5+length/2-0.3, i-0.5+width/2-0.1, int(j), fontsize = 7)
        plt.axvline(x=j+1-0.5, ymin=0, ymax=Shift_Number, color = "black")
    plt.axhline(y=i+1-0.5, xmin=0, xmax=Staff_Number, color = "black")
plt.xlabel("Staff")
plt.ylabel("Shift")
plt.xlim([0.5,Staff_Number+0.5])
plt.ylim([0.5,Shift_Number+0.5])
