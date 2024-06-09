// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "FiringRangeTarget.h"
#include "FiringRangeDummy.generated.h"

UCLASS()
class FIRINGRANGERUNTIME_API AFiringRangeDummy : public AFiringRangeTarget
{
	GENERATED_BODY()

public:
	// Sets default values for this character's properties
	AFiringRangeDummy(const FObjectInitializer& ObjectInitializer = FObjectInitializer::Get());

	virtual void InitializeAbilitySystem() override;
	virtual void BeginPlay() override;

private:
	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "Mesh", meta = (AllowPrivateAccess = "true"))
	UStaticMeshComponent* TargetMesh;

	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "Mesh", meta = (AllowPrivateAccess = "true"))
	UStaticMeshComponent* BaseMesh;

	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "Mesh", meta = (AllowPrivateAccess = "true"))
	UStaticMeshComponent* TargetPoint;
	
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Rotation", meta = (AllowPrivateAccess = "true"))
	bool bIsDead;

public:
	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "Rotation", meta = (AllowPrivateAccess = "true"))
	FRotator StartRotation;

	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "Rotation", meta = (AllowPrivateAccess = "true"))
	FRotator EndRotation;

	UFUNCTION(BlueprintImplementableEvent, BlueprintCallable, Category = "Rotation")
	void RotateTarget(FRotator InStartRotation, FRotator InEndRotation);

	UStaticMeshComponent* GetTargetPointMesh() const { return TargetPoint; }
};
