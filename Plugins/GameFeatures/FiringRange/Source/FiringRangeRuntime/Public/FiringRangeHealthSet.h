// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "AbilitySystem/Attributes/LyraHealthSet.h"
#include "FiringRangeHealthSet.generated.h"

/**
 * 
 */
UCLASS()
class FIRINGRANGERUNTIME_API UFiringRangeHealthSet : public ULyraHealthSet
{
	GENERATED_BODY()

public:
	UFiringRangeHealthSet();

	virtual void PreAttributeChange(const FGameplayAttribute& Attribute, float& NewValue) override;
	virtual void PreAttributeBaseChange(const FGameplayAttribute& Attribute, float& NewValue) const override;
};
