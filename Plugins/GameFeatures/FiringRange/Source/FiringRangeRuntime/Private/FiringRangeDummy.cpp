// Fill out your copyright notice in the Description page of Project Settings.

#include "FiringRangeDummy.h"
#include "FiringRangeHealthSet.h"
#include "AbilitySystem/Attributes/LyraHealthSet.h"

// Sets default values
AFiringRangeDummy::AFiringRangeDummy(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer
		.SetDefaultSubobjectClass<UFiringRangeHealthSet>("HealthSet"))
{
	TargetMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("Target Mesh"));
	TargetMesh->SetupAttachment(GetMesh());

	BaseMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("Base Mesh"));
	BaseMesh->SetupAttachment(GetMesh());

	TargetPoint = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("Target Point"));
	TargetPoint->SetupAttachment(TargetMesh);

	bIsDead = true;
}

void AFiringRangeDummy::InitializeAbilitySystem()
{
	Super::InitializeAbilitySystem();

	// Initialize default values for Health Set
	HealthSet->InitMaxHealth(1.f);
	HealthSet->InitHealth(1.f);
}

void AFiringRangeDummy::BeginPlay()
{
	Super::BeginPlay();

	// Set StartRotation to be the current rotation of the TargetMesh (upright)
	StartRotation = TargetMesh->GetComponentRotation();

	// Set EndRotation to be 90 degrees from StartRotation
	EndRotation = StartRotation + FRotator(0.f, 0.f, 90.f);
}
