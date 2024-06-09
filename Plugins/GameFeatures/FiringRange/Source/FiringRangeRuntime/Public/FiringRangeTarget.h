// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

// ALyraCharacter needs "LyraGame", "GameplayAbilities" and "AIModule" in Build.cs

#include "CoreMinimal.h"
#include "Character/LyraCharacter.h"
#include "FiringRangeTarget.generated.h"

class ULyraAbilitySet;
class UAbilitySystemComponent;
class ULyraAbilitySystemComponent;
class ULyraCombatSet;
class ULyraHealthSet;

UCLASS()
class FIRINGRANGERUNTIME_API AFiringRangeTarget : public ALyraCharacter
{
	GENERATED_BODY()

	// Ability system component sub-object used by player characters
	UPROPERTY(VisibleAnywhere, Category = "FR|Ability")
	TObjectPtr<ULyraAbilitySystemComponent> AbilitySystemComponent;

	// AbilitySet this actor should be granted on spawn, if any
	UPROPERTY(EditDefaultsOnly, Category="FR|Ability")
	TObjectPtr<ULyraAbilitySet> AbilitySetOnSpawn;

public:
	// Sets default values for this character's properties
	AFiringRangeTarget(const FObjectInitializer& ObjectInitializer = FObjectInitializer::Get());

	// IAbilitySystemComponent Interface
	virtual UAbilitySystemComponent* GetAbilitySystemComponent() const override;
	ULyraAbilitySystemComponent* GetLyraAbilitySystemComponentChecked() const;

protected:
	// Combat Set for healing and damage calculations. Defines health and damage
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="FR|Ability")
	TObjectPtr<ULyraCombatSet> CombatSet;

	// Health Set for healing and damage calculations. Defines health, max health, incoming healing and incoming damage
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="FR|Ability")
	TObjectPtr<ULyraHealthSet> HealthSet;

	// Initialization and cleanup
	virtual void PostInitializeComponents() override;
	virtual void InitializeAbilitySystem();
	virtual void UninitializeAbilitySystem();
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
};
