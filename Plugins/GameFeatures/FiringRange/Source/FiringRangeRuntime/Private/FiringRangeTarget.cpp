// Fill out your copyright notice in the Description page of Project Settings.

#include "FiringRangeTarget.h"
#include "AbilitySystemComponent.h"
#include "AbilitySystem/LyraAbilitySystemComponent.h"
#include "AbilitySystem/Attributes/LyraCombatSet.h"
#include "AbilitySystem/Attributes/LyraHealthSet.h"

// Sets default values
AFiringRangeTarget::AFiringRangeTarget(const FObjectInitializer& ObjectInitializer) : Super(ObjectInitializer)
{
	// Set this character to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = false;

	AbilitySystemComponent = ObjectInitializer.CreateDefaultSubobject<ULyraAbilitySystemComponent>(
		this, TEXT("AbilitySystemComponent"));
	AbilitySystemComponent->SetIsReplicated(true);
	AbilitySystemComponent->SetReplicationMode(EGameplayEffectReplicationMode::Mixed);

	HealthSet = ObjectInitializer.CreateDefaultSubobject<ULyraHealthSet>(this, TEXT("HealthSet"));
	CombatSet = ObjectInitializer.CreateDefaultSubobject<ULyraCombatSet>(this, TEXT("CombatSet"));
}

void AFiringRangeTarget::PostInitializeComponents()
{
	// BEFORE PostInit Components:
	InitializeAbilitySystem();

	// Now during PostInit Components, there is a functional ASC for other components to use
	Super::PostInitializeComponents();
}

// Called when the game starts or when spawned
void AFiringRangeTarget::BeginPlay()
{
	Super::BeginPlay();

	if (HasAuthority() && AbilitySetOnSpawn)
	{
		// AbilitySetOnSpawn->GiveToAbilitySystem(AbilitySystemComponent, &GrantedHandlesOnSpawn);
	}
}

void AFiringRangeTarget::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	UninitializeAbilitySystem();

	Super::EndPlay(EndPlayReason);
}

void AFiringRangeTarget::InitializeAbilitySystem()
{
	// We expect this to have been set in the constructor
	check(IsValid(AbilitySystemComponent));

	// Initialize ASC on this Actor
	AbilitySystemComponent->InitAbilityActorInfo(this, this);

	// Add Attribute Sets to ASC
	AbilitySystemComponent->AddAttributeSetSubobject(CombatSet.Get());
	AbilitySystemComponent->AddAttributeSetSubobject(HealthSet.Get());

	// DO NOT init HealthComponent until AFTER HealthSet has been added
	OnAbilitySystemInitialized();
}


void AFiringRangeTarget::UninitializeAbilitySystem()
{
	OnAbilitySystemUninitialized();

	if (AbilitySystemComponent)
	{
		AbilitySystemComponent->RemoveSpawnedAttribute(CombatSet.Get());
		AbilitySystemComponent->RemoveSpawnedAttribute(HealthSet.Get());

		AbilitySystemComponent->CancelAbilities();
		AbilitySystemComponent->ClearAbilityInput();
		AbilitySystemComponent->RemoveAllGameplayCues();
		AbilitySystemComponent->ClearActorInfo();
		// GrantedHandlesOnSpawn.TakeFromAbilitySystem(AbilitySystemComponent);
	}
}


UAbilitySystemComponent* AFiringRangeTarget::GetAbilitySystemComponent() const
{
	return AbilitySystemComponent;
}

ULyraAbilitySystemComponent* AFiringRangeTarget::GetLyraAbilitySystemComponentChecked() const
{
	check(AbilitySystemComponent);
	return AbilitySystemComponent;
}
