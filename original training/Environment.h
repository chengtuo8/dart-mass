#ifndef __MASS_ENVIRONMENT_H__
#define __MASS_ENVIRONMENT_H__
#include "dart/dart.hpp"
#include "Character.h"
#include "Muscle.h"
namespace MASS
{

struct MuscleTuple
{
	// Eigen::VectorXd 一个长度为N的double类型向量
	Eigen::VectorXd JtA; //肌肉力转为关节力矩的导数矩阵 A
	Eigen::VectorXd L;// 加速度关于肌肉激活的线性系数
	Eigen::VectorXd b;//肌肉单位的偏置
	Eigen::VectorXd tau_des;//期望的关节加速度
};
class Environment
{
public:
	Environment(); //构造函数声明

	// 初始化过程
	void SetUseMuscle(bool use_muscle){mUseMuscle = use_muscle;}
	void SetControlHz(int con_hz) {mControlHz = con_hz;}
	void SetSimulationHz(int sim_hz) {mSimulationHz = sim_hz;}

	void SetCharacter(Character* character) {mCharacter = character;}
	void SetGround(const dart::dynamics::SkeletonPtr& ground) {mGround = ground;}

	void SetRewardParameters(double w_q,double w_v,double w_ee,double w_com){this->w_q = w_q;this->w_v = w_v;this->w_ee = w_ee;this->w_com = w_com;}
	// 初始化，加载模型结构，肌肉拓扑等数据
	void Initialize();
	void Initialize(const std::string& meta_file,bool load_obj = false);
public:
	void Step(); // 推进一次仿真
	void Reset(bool RSI = true); //环境重置
	bool IsEndOfEpisode(); // 判断当前episode是否终止
	Eigen::VectorXd GetState();// 输出当前状态向量s
	void SetAction(const Eigen::VectorXd& a);//接收上一个网络输出的动作向量a
	double GetReward();//计算奖励函数

	Eigen::VectorXd GetDesiredTorques();//生成期望力矩
	Eigen::VectorXd GetMuscleTorques();//生成肌肉合力矩

	const dart::simulation::WorldPtr& GetWorld(){return mWorld;} //返回仿真世界的指针
	Character* GetCharacter(){return mCharacter;} // 返回角色指针
	const dart::dynamics::SkeletonPtr& GetGround(){return mGround;} //返回地面指针

	int GetControlHz(){return mControlHz;}
	int GetSimulationHz(){return mSimulationHz;}
	int GetNumTotalRelatedDofs(){return mCurrentMuscleTuple.JtA.rows();} //返回自由度
	std::vector<MuscleTuple>& GetMuscleTuples(){return mMuscleTuples;};//返回一个state
	int GetNumState(){return mNumState;}
	int GetNumAction(){return mNumActiveDof;}
	int GetNumSteps(){return mSimulationHz/mControlHz;}
	
	const Eigen::VectorXd& GetActivationLevels(){return mActivationLevels;}// 返回肌肉激活向量
	const Eigen::VectorXd& GetAverageActivationLevels(){return mAverageActivationLevels;}//返回平均激活程度
	void SetActivationLevels(const Eigen::VectorXd& a){mActivationLevels = a;}// 设置当前肌肉激活的向量
	bool GetUseMuscle(){return mUseMuscle;} //是否使用肌肉模型

	// —— 控制层（整场运动）的功率/能量（新增）——
	double GetEpisodeCtrlEnergy() const { 
		return mCtrlEnergyEpisode; 
	}
	double GetEpisodeCtrlAvgPower() const {
		return (mCtrlPowerCount>0)? (mCtrlPowerSum/mCtrlPowerCount) : 0.0;
	}

private:
	dart::simulation::WorldPtr mWorld;//仿真世界指针
	int mControlHz,mSimulationHz;
	bool mUseMuscle;
	Character* mCharacter;
	dart::dynamics::SkeletonPtr mGround;
	Eigen::VectorXd mAction;//当前动作向量，启用肌肉输出肌肉激活，不启用输出期望加速度
	Eigen::VectorXd mTargetPositions,mTargetVelocities; 

	int mNumState;
	int mNumActiveDof;
	int mRootJointDof;

	Eigen::VectorXd mActivationLevels;
	Eigen::VectorXd mAverageActivationLevels;
	Eigen::VectorXd mDesiredTorque;
	std::vector<MuscleTuple> mMuscleTuples;
	MuscleTuple mCurrentMuscleTuple;
	int mSimCount;
	int mRandomSampleIndex;

	// —— 控制层（整场运动）的功率/能量（新增）——
	double mCtrlEnergyEpisode = 0.0;  // ∫ |τ_des · qdot| dt
	double mCtrlPowerSum = 0.0;       // 累加 |P|
	int    mCtrlPowerCount = 0;       // 计数


	double w_q,w_v,w_ee,w_com;
};
};

#endif