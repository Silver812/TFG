// Fill out your copyright notice in the Description page of Project Settings.

#include "FiringRangePlayerController.h"
#include "AbilitySystemComponent.h"
#include "AbilitySystemGlobals.h"
#include "AbilitySystemLog.h"
#include "FiringRangeDummy.h"
#include "SIOJsonValue.h"
#include "SIOJConvert.h"
#include "SocketIOClientComponent.h"
#include "GameModes/LyraGameMode.h"
#include "Kismet/GameplayStatics.h"

// Sets default values
AFiringRangePlayerController::AFiringRangePlayerController(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer)
{
	SocketIOClientComponent = CreateDefaultSubobject<USocketIOClientComponent>(TEXT("SocketIOClientComponent"));

	// Disable auto connect
	SocketIOClientComponent->bShouldAutoConnect = false;
	
	// Initialize the struct
	RLEngineStruct = FRLEngineStruct(TEXT("Observation"), 0.0f, false, false, 0.0f);
}


void AFiringRangePlayerController::BeginPlay()
{
	Super::BeginPlay();

	// Bind events
	SocketIOClientComponent->OnConnected.AddDynamic(this, &AFiringRangePlayerController::OnConnected);
	
	FiringRangeModeRef = Cast<ALyraGameMode>(UGameplayStatics::GetGameMode(GetWorld()));

	// Edit the values of the struct in B_LyraGameMode
	RLEngineStruct = FiringRangeModeRef->RLEngineStruct;
	
	if (!FiringRangeModeRef)
	{
		GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Red, TEXT("FiringRangeMode not found"));
	}
	
	// Get viewport size
	FVector2D ViewportSize;
	GEngine->GameViewport->GetViewportSize(ViewportSize);

	RLEngineStruct.SizeX = ViewportSize.X;
	RLEngineStruct.SizeY = ViewportSize.Y;

	PixelDistanceXY = ViewportSize * RLEngineStruct.DistancePercentage;

	// Debug distance X and Y
	// GEngine->AddOnScreenDebugMessage(-1, 9999999.f, FColor::Blue, FString::Printf(TEXT("Distance %lf"), (PixelDistanceXY.X + PixelDistanceXY.Y)*0.6));
}

void AFiringRangePlayerController::OnConnected(FString SocketId, FString SessionId, bool bIsReconnection)
{
	SocketIOClientComponent->OnNativeEvent(
		TEXT("send_log"), [this](const FString& Event, const TSharedPtr<FJsonValue>& Message)
		{
			this->OnLogReceived(Event, Message);
		});

	SocketIOClientComponent->OnNativeEvent(
		TEXT("send_action"), [this](const FString& Event, const TSharedPtr<FJsonValue>& Message)
		{
			this->OnActionReceived(Event, Message);
		});

	SocketIOClientComponent->OnNativeEvent(
		TEXT("receive_reset_data"), [this](const FString& Event, const TSharedPtr<FJsonValue>& Message)
		{
			this->OnResetDataRequest(Event, Message);
		});

	LaunchRLEngine();
}

void AFiringRangePlayerController::RLStructToEncodedJson()
{
	TSharedPtr<FJsonObject> JsonObject = USIOJConvert::ToJsonObject(FRLEngineStruct::StaticStruct(), &RLEngineStruct);
	SIOJsonObject = NewObject<USIOJsonObject>();
	SIOJsonObject->SetRootObject(JsonObject);
	const FString Message = SIOJsonObject->EncodeJson();
	TSharedPtr<FJsonValue> FJsonValue = MakeShareable(new FJsonValueString(Message));
	SIOJsonValue = NewObject<USIOJsonValue>();
	SIOJsonValue->SetRootValue(FJsonValue);
}

void AFiringRangePlayerController::LaunchRLEngine()
{
	RLStructToEncodedJson();
	SocketIOClientComponent->Emit(TEXT("launch_rl_engine"), SIOJsonValue);

	Agent = GetPawn();
}

void AFiringRangePlayerController::InitRLEngineStruct(const FRLEngineStruct& InRLEngineStruct)
{
	RLEngineStruct = InRLEngineStruct;
}

void AFiringRangePlayerController::OnLogReceived(const FString& Event, const TSharedPtr<FJsonValue>& Message)
{
	GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Green, Message->AsString());
}

void AFiringRangePlayerController::OnActionReceived(const FString& Event, const TSharedPtr<FJsonValue>& Message)
{
	EActions Action = static_cast<EActions>(static_cast<int>(Message->AsNumber()));

	// Debug action
	// GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Green, ActionToString[Action]);
	
	if (Action == ShootAction)
	{
		SendObservation();
		AddGameplayEvent(FName("InputTag.Weapon.Fire"));
	}
	else
	{
		FVector2D Direction = FVector2d(RLEngineStruct.SizeX/2, RLEngineStruct.SizeY/2) - (PixelDistanceXY * ActionToDirection[Action]);
		if(GetTargetScreenLocation(TargetScreenLocation))
		{
			FVector2D NextScreenLocation = Direction;
			FVector WorldPosition, WorldDirection;
			DeprojectScreenPositionToWorld(NextScreenLocation.X, NextScreenLocation.Y, WorldPosition, WorldDirection);

			// Distance from the camera (WorldPosition) to the target
			const double Distance = FVector::Dist(WorldPosition, FiringRangeModeRef->TargetLocation);
			NextLocation = WorldPosition + WorldDirection * Distance;

			// Draw a debug sphere
			DrawDebugSphere(GetWorld(), NextLocation, 20.0f, 12, FColor::Red, false, 0.5f, 1, 1.f);

			if (APawn* ControlledPawn=GetPawn())
			{
				// Cast to ALyraCharacter
				if (ALyraCharacter* LyraCharacter = Cast<ALyraCharacter>(ControlledPawn))
				{
					// Move the player camera smoothly
					LyraCharacter->MovePlayerCameraSmooth(GetControlRotation(), 0.1, NextLocation);
				}
			}
		}
		else
		{
			// If the target is not on the screen, the observation is invalid
			RLEngineStruct.Observation = "";
			GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Green, TEXT("Target is not on the screen"));
		}
	}
}

void AFiringRangePlayerController::OnResetDataRequest(const FString& Event, const TSharedPtr<FJsonValue>& Message)
{
	// Debug initial message
	// GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Green, Message->AsString());
	SendObservation();
}

bool AFiringRangePlayerController::MakeObservation()
{	
	// Get the target position in the world
	const bool bIsOnScreen = GetTargetScreenLocation(TargetScreenLocation);
	
	// Transform the Vector2D into a string
	RLEngineStruct.Observation = Vector2DToString(TargetScreenLocation);

	// Debug target screen location
	// FVector2D TempScreenLocation = TargetScreenLocation - FVector2D(RLEngineStruct.SizeX/2, RLEngineStruct.SizeY/2);
	// GEngine->AddOnScreenDebugMessage(-1, 1.f, FColor::Green, FString::Printf(TEXT("%lf"), (FMath::Abs(TempScreenLocation.X) + FMath::Abs(TempScreenLocation.Y))*0.6));
	// GEngine->AddOnScreenDebugMessage(-1, 1.f, FColor::Red, FString::Printf(TEXT("%lf, %lf"), TempScreenLocation.X, TempScreenLocation.Y));
	
	return bIsOnScreen;
}

void AFiringRangePlayerController::AddGameplayEvent(const FName& GameplayTag) const
{
	if (IsValid(Agent))
	{
		const FGameplayTag EventTag = FGameplayTag::RequestGameplayTag(GameplayTag);

		UAbilitySystemComponent* AbilitySystemComponent =
			UAbilitySystemGlobals::GetAbilitySystemComponentFromActor(Agent);
		if (AbilitySystemComponent != nullptr && IsValidChecked(AbilitySystemComponent))
		{
			FScopedPredictionWindow NewScopedWindow(AbilitySystemComponent, true);
			const FGameplayEventData Payload;
			AbilitySystemComponent->HandleGameplayEvent(EventTag, &Payload);
		}
		else
		{
			ABILITY_LOG(
				Error,
				TEXT(
					"FiringRangePlayerController::AddGameplayEvent: Invalid ability system component retrieved from Actor %s. EventTag was %s"
				), *Agent->GetName(), *EventTag.ToString());
		}
	}
	else
	{
		GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Red, TEXT("Invalid agent"));
	}
}

FString AFiringRangePlayerController::Vector2DToString(const FVector2D& Vector)
{
	return FString::Printf(TEXT("%lf, %lf"), Vector.X, Vector.Y);
}

bool AFiringRangePlayerController::GetTargetScreenLocation(FVector2D& OutScreenLocation) const
{
	if (FiringRangeModeRef)
	{
		FiringRangeModeRef->UpdateTargetLocation();
	}

	// Debug sphere
	DrawDebugSphere(GetWorld(), FiringRangeModeRef->TargetLocation, 20.0f, 12, FColor::Green, false, 0.5f, 1, 1.f);

	// Get the target position in the viewport
	return ProjectWorldLocationToScreen(FiringRangeModeRef->TargetLocation, OutScreenLocation);
}

void AFiringRangePlayerController::SendObservation()
{
	MakeObservation();
	RLStructToEncodedJson();
	SocketIOClientComponent->Emit(TEXT("send_obs"), SIOJsonValue);
}

void AFiringRangePlayerController::SendScore(int32 InScore)
{
	SendMessage(FString::Printf(TEXT("Score: %d"), InScore));
}

void AFiringRangePlayerController::SendMessage(const FString& InMessage)
{
	RLEngineStruct.Message = InMessage;
	RLStructToEncodedJson();
	SocketIOClientComponent->Emit(TEXT("send_message"), SIOJsonValue);
}

void AFiringRangePlayerController::OnPossess(APawn* InPawn)
{
	Super::OnPossess(InPawn);

	FiringRangeModeRef->NextObjective();

	SocketIOClientComponent->Connect(TEXT("http://localhost:3000"));
}
