// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Player/LyraPlayerController.h"
#include "FiringRangePlayerController.generated.h"

class ALyraGameMode;
class AFiringRangeMode;

USTRUCT(BlueprintType, Blueprintable)
struct FRLEngineStruct
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RLEngine")
	FString Observation;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RLEngine")
	float Reward;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RLEngine")
	bool bTerminated;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RLEngine")
	bool bTruncated;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RLEngine")
	float Info;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RLEngine")
	int32 SizeX;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RLEngine")
	int32 SizeY;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RLEngine")
	int32 MaxTimesteps;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RLEngine")
	double DistancePercentage;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RLEngine")
	int32 EvalEpisodes;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RLEngine")
	float UpdateTime;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RLEngine")
	FString RLAlgorithm; // ppo, ppo_ext or a2c

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RLEngine")
	FString Message;
	
	FRLEngineStruct() = default;

	FRLEngineStruct(const FString& InObservation, const float InReward, const bool bInTerminated,
	                const bool bInTruncated, const float InInfo, const int32 InSizeX = 1920, const int32 InSizeY = 1080,
	                const int32 InMaxTimesteps = 50000, const double InDistancePercentage = 0.03f,
	                const int32 InEvalEpisodes = 15, const float InUpdateTime = .001f, const FString& InRLAlgorithm = TEXT("ppo"), const FString& InMessage = TEXT("")):
		Observation(InObservation),
		Reward(InReward),
		bTerminated(bInTerminated),
		bTruncated(bInTruncated),
		Info(InInfo),
		SizeX(InSizeX),
		SizeY(InSizeY),
		MaxTimesteps(InMaxTimesteps),
		DistancePercentage(InDistancePercentage),
		EvalEpisodes(InEvalEpisodes),
		UpdateTime(InUpdateTime),
		RLAlgorithm(InRLAlgorithm),
		Message(InMessage)
	{
	}
};

UENUM()
enum EActions
{
	RightAction,
	UpAction,
	LeftAction,
	DownAction,
	ShootAction,
	RightUpAction,
	RightDownAction,
	LeftUpAction,
	LeftDownAction
};

class USIOJsonValue;
class USIOJsonObject;
class USocketIOClientComponent;

/**
 * 
 */
UCLASS()
class FIRINGRANGERUNTIME_API AFiringRangePlayerController : public ALyraPlayerController
{
	GENERATED_BODY()

	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "SocketIO", meta=(AllowPrivateAccess="true"))
	USocketIOClientComponent* SocketIOClientComponent;

	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "SocketIO", meta=(AllowPrivateAccess="true"))
	USIOJsonObject* SIOJsonObject;

	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "SocketIO", meta=(AllowPrivateAccess="true"))
	USIOJsonValue* SIOJsonValue;

	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "RLEngine", meta=(AllowPrivateAccess="true"))
	FRLEngineStruct RLEngineStruct;

	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "RLEngine", meta=(AllowPrivateAccess="true"))
	AActor* Agent;

	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "RLEngine", meta=(AllowPrivateAccess="true"))
	FVector2D PixelDistanceXY;
	
	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "RLEngine", meta=(AllowPrivateAccess="true"))
	double CurrentX;

	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "RLEngine", meta=(AllowPrivateAccess="true"))
	double CurrentY;
	
	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "RLEngine", meta=(AllowPrivateAccess="true"))
	double LastX;

	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "RLEngine", meta=(AllowPrivateAccess="true"))
	double LastY;
	
	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "RLEngine", meta=(AllowPrivateAccess="true"))
	FVector NextLocation;

	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "RLEngine", meta=(AllowPrivateAccess="true"))
	FVector2D TargetScreenLocation;

	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "RLEngine", meta=(AllowPrivateAccess="true"))
	ALyraGameMode* FiringRangeModeRef;
	
	// Converts a movement action into a direction
	TMap<EActions, FVector2D> ActionToDirection
	{
		{RightAction, FVector2D(1.f, 0.f)},
		{UpAction, FVector2D(0.f, 1.f)},
		{LeftAction, FVector2D(-1.f, 0.f)},
		{DownAction, FVector2D(0.f, -1.f)},
		{RightUpAction, FVector2D(1.f, 1.f)},
		{RightDownAction, FVector2D(1.f, -1.f)},
		{LeftUpAction, FVector2D(-1.f, 1.f)},
		{LeftDownAction, FVector2D(-1.f, -1.f)}
	};

	// Converts an action to a string
	TMap<EActions, FString> ActionToString
	{
		{RightAction, "Right"},
		{UpAction, "Up"},
		{LeftAction, "Left"},
		{DownAction, "Down"},
		{ShootAction, "Shoot"},
		{RightUpAction, "RightUp"},
		{RightDownAction, "RightDown"},
		{LeftUpAction, "LeftUp"},
		{LeftDownAction, "LeftDown"}
	};

public:
	// Sets default values for this actor's properties
	AFiringRangePlayerController(const FObjectInitializer& ObjectInitializer = FObjectInitializer::Get());

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:
	UFUNCTION()
	void OnConnected(FString SocketId, FString SessionId, bool bIsReconnection);

	// Converts the RLEngineStruct to a JSON object and then encodes it to a string-like SIOJsonValue
	void RLStructToEncodedJson();

	// Launches the RLEngine by emitting the RLEngineStruct as a JSON object
	UFUNCTION()
	void LaunchRLEngine();

	UFUNCTION(BlueprintCallable, Category = "FiringRange")
	void InitRLEngineStruct(const FRLEngineStruct& InRLEngineStruct);
	
	// Receives a log or order from the python server
	void OnLogReceived(const FString& Event, const TSharedPtr<FJsonValue>& Message);

	// Receives an action from the agent
	void OnActionReceived(const FString& Event, const TSharedPtr<FJsonValue>& Message);

	// Sends initial observation and info to the agent
	void OnResetDataRequest(const FString& Event, const TSharedPtr<FJsonValue>& Message);

	// Sends an observation to the agent
	UFUNCTION()
	bool MakeObservation();

	// Sends a gameplay event to the controlled actor
	void AddGameplayEvent(const FName& GameplayTag) const;

	// Transforms a Vector2D into a FString
	static FString Vector2DToString(const FVector2D& Vector);

	bool GetTargetScreenLocation(FVector2D& OutScreenLocation) const;

	// Send and observation to the agent
	UFUNCTION(BlueprintCallable, Category = "FiringRange")
	void SendObservation();

	UFUNCTION(BlueprintCallable, Category = "FiringRange")
	void SendScore(int32 InScore);

	UFUNCTION(BlueprintCallable, Category = "FiringRange")
	void SendMessage(const FString& InMessage);
	
	virtual void OnPossess(APawn* InPawn) override;
};
