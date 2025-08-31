#ifndef __MASS_ENVIRONMENT_H__
#define __MASS_ENVIRONMENT_H__
#include "dart/dart.hpp"
#include "Character.h"
#include "Muscle.h"
namespace MASS
{

struct MuscleTuple
{
	Eigen::VectorXd JtA;
	Eigen::VectorXd L;
	Eigen::VectorXd b;
	Eigen::VectorXd tau_des;
};
class Environment
{
public:
	Environment();

	void SetUseMuscle(bool use_muscle){mUseMuscle = use_muscle;}
	void SetControlHz(int con_hz) {mControlHz = con_hz;}
	void SetSimulationHz(int sim_hz) {mSimulationHz = sim_hz;}

	void SetCharacter(Character* character) {mCharacter = character;}
	void SetGround(const dart::dynamics::SkeletonPtr& ground) {mGround = ground;}

	void SetRewardParameters(double w_q,double w_v,double w_ee,double w_com){this->w_q = w_q;this->w_v = w_v;this->w_ee = w_ee;this->w_com = w_com;}
	void Initialize();
	void Initialize(const std::string& meta_file,bool load_obj = false);
public:
	void Step();
	void Reset(bool RSI = true);
	bool IsEndOfEpisode();
	Eigen::VectorXd GetState();
	void SetAction(const Eigen::VectorXd& a);
	double GetReward();

	Eigen::VectorXd GetDesiredTorques();
	Eigen::VectorXd GetMuscleTorques();

	// 外骨骼接口（6维）
	void SetUseExo(bool use){ mUseExo = use; }
	bool GetUseExo() const { return mUseExo; }
	void SetExoTorqueLimits(const Eigen::VectorXd& lim);      // size = 6
	void SetExoTorques(const Eigen::VectorXd& tau_exo6);      // size = 6
	const Eigen::VectorXd& GetExoTorques() const { return mExoTau6; }
	double GetEpisodeExoEnergy() const { return mExoEnergyEpisode; }
	double GetEpisodeExoAvgPower() const { return (mExoPowerCount>0)? (mExoPowerSum/mExoPowerCount) : 0.0; }
	void   ResetExoEpisodeAccumulators();


	//控制层能量/功率统计（新增）
	double GetEpisodeCtrlEnergy() const { 
    return mCtrlEnergyEpisode; 
	}
	double GetEpisodeCtrlAvgPower() const {
		return (mCtrlPowerCount>0)? (mCtrlPowerSum/mCtrlPowerCount) : 0.0;
	}

	const dart::simulation::WorldPtr& GetWorld(){return mWorld;}
	Character* GetCharacter(){return mCharacter;}
	const dart::dynamics::SkeletonPtr& GetGround(){return mGround;}
	int GetControlHz(){return mControlHz;}
	int GetSimulationHz(){return mSimulationHz;}
	int GetNumTotalRelatedDofs(){return mCurrentMuscleTuple.JtA.rows();}
	std::vector<MuscleTuple>& GetMuscleTuples(){return mMuscleTuples;};
	int GetNumState(){return mNumState;}
	int GetNumAction(){return mNumActiveDof;}
	int GetNumSteps(){return mSimulationHz/mControlHz;}
	
	const Eigen::VectorXd& GetActivationLevels(){return mActivationLevels;}
	const Eigen::VectorXd& GetAverageActivationLevels(){return mAverageActivationLevels;}
	void SetActivationLevels(const Eigen::VectorXd& a){mActivationLevels = a;}
	bool GetUseMuscle(){return mUseMuscle;}
private:
	dart::simulation::WorldPtr mWorld;
	int mControlHz,mSimulationHz;
	bool mUseMuscle;
	Character* mCharacter;
	dart::dynamics::SkeletonPtr mGround;
	Eigen::VectorXd mAction;
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

	// —— 外骨骼（仅双髋 6 维）——
	bool mUseExo = true;               // 是否启用外骨骼
	std::vector<int> mExoActiveIdx;     // 映射到 active dofs 的下标（长度应为 6）
	Eigen::VectorXd mExoTau6;           // 外骨骼 6 维力矩（左右髋各3轴）
	Eigen::VectorXd mExoTauLimit6;      // 6 维上限
	Eigen::VectorXd mExoTauAct;         // 尺寸 = mNumActiveDof，用来散射到有效关节（仅在 Step() 内部使用）

	double mExoEnergyEpisode = 0.0;     // 本回合外骨骼累计能量(∫|P|dt)
	double mExoPowerSum = 0.0;          // 本回合功率的绝对值累计，用于计算平均功率
	int    mExoPowerCount = 0;          // 计数


		// —— 控制层（整场运动）的功率/能量（新增）——
	double mCtrlEnergyEpisode = 0.0;  // ∫ |τ_des · qdot| dt
	double mCtrlPowerSum = 0.0;       // 累加 |P|
	int    mCtrlPowerCount = 0;       // 计数

	double w_q,w_v,w_ee,w_com;
};
};

#endif