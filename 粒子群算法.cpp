#include<iostream>
#include<math.h>
using namespace std;
#define csize 20//种群的规模
#define c1 1.5
#define c2 1.5//学习因子
#define dim 2//维数
#define v_max 1
#define v_min -1//定义种群的速度极值
#define stri_max 10
#define stri_min -10//定义极大极小值
#define max_try 100
#define w 1//惯性因子 1

double pop_home[csize][dim];//定义种群的大小
double pop_v[csize][dim];//种群个体的速度
double pop_value[csize];//存放种群的适应度
double best_fitnum[max_try];//存放每次的最优适应值
double all_best_pos[dim];//记录群体极值的位置
double all_best_num;//记录群体极值适应度值
double pop_best_num[csize];//记录个体的适应度极值
double pop_best_pos[csize][dim];//记录个体适应度极值的位置
double gen_best[max_try][dim];//记录每一代的极值位置
double result[max_try];//记录每一代的最值
double func(double x, double y) 
{
	
	return -((x * x) - (y * y));//适应度函数

}



void init() //随机初始化种群个体
{
	for (int i = 0; i < csize; i++) {
		for (int j = 0; j < dim; j++) {
			pop_home[i][j] = (((double)rand()) / RAND_MAX - 0.5) * 20;
			pop_v[i][j]= (((double)rand()) / RAND_MAX - 0.5) * 2;
			pop_value[i] = func(pop_home[i][j], pop_v[i][j]);
		}
		
	}//完成初始化，以及适应度值的初始指定









}
double* find_max(double* fit,int size) //找到适应度极值，以及所对应的位置
{
	int num = 0;
	double max = *fit;
	double best[2];
	for (int i = 1; i < size; i++) {
		if (*(fit + i) > max) {
			num = i;
			max = *(fit + i);
		}
	}
	best[0] =num;
	best[1] = max;
	return best;
}


void PSO()//粒子群算法求极值
{
	init();//
	double* bestfit = find_max(pop_value, csize);//找到种群的最大位置
	int index = int(*bestfit);//记录群体极值的编号
	for (int i = 0; i < dim; i++) {
		all_best_pos[i] = pop_home[index][i];//记录群体极值的各维度的坐标记录下来
	}
	all_best_num = (*bestfit + 1);//保存群体极值的大小





	for (int i = 0; i < csize; i++) {
		pop_best_num[i] = pop_value[i];//记录个体达到的最值
		for (int j = 0; j < dim; j++) {
			pop_best_pos[i][j] = pop_home[i][j];//记录个体达到极值的坐标
		}
	}
	for (int i = 0; i < max_try; i++) 
	{
		for (int j = 0; j < csize; j++) {
			//首先进行个体的速度以及位置更新
			for (int k = 0; k < dim; k++)
			{
				double rand1 = (double)rand() / RAND_MAX; 
				double rand2 = (double)rand() / RAND_MAX;
				pop_v[j][k] = w * pop_v[j][k] + c1 * rand1 * (pop_best_pos[j][k] - pop_home[j][k]) + c2 * rand2 * (all_best_pos[k] - pop_home[j][k]);
				if (pop_v[j][k] > v_max)pop_v[j][k] = v_max;
				if (pop_v[j][k] < v_min)pop_v[j][k] = v_min;
				//位置更新
				pop_home[j][k] += pop_v[j][k];
				if (pop_home[j][k] > stri_max)pop_home[j][k] =stri_max;
				if (pop_home[j][k] <stri_min)pop_home[j][k] = stri_min;
			}
           
			pop_value[j] = func(pop_home[j][0], pop_home[j][1]);

			//进行个体极值的更新
			if (pop_value[j] > pop_best_num[j]) {
				for (int j2 = 0; j2 < dim; j2++) {
					pop_best_pos[j][j2] = pop_home[j][j2];

				}
				pop_best_num[j] = pop_value[j];
			}
			if (pop_value[j] > all_best_num) 
			{
				for (int j2 = 0; j2 < dim; j2++) {
					all_best_pos[j2] = pop_home[j][j2];
				}
				all_best_num = pop_value[j];
			}
			



		}
		for (int k1 = 0; k1 < dim; k1++) 
		{
			gen_best[i][k1] = all_best_pos[k1];
		}
		result[i] = all_best_num;
		cout << endl;
		cout << "第" << i << "次迭代极值为" << -all_best_num << endl;
		cout << "取得极值的坐标为" << all_best_pos[0] << "  " << all_best_pos[1] << endl;

	}

}
int main() {
	PSO();//运行计算最值
	return 0;





}