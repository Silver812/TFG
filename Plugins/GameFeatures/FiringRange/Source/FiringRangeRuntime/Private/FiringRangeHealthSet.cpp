// Fill out your copyright notice in the Description page of Project Settings.

#include "FiringRangeHealthSet.h"

UFiringRangeHealthSet::UFiringRangeHealthSet() : Super()
{
}

void UFiringRangeHealthSet::PreAttributeChange(const FGameplayAttribute& Attribute, float& NewValue)
{
	Super::PreAttributeChange(Attribute, NewValue);

	if (Attribute == GetHealthAttribute() && NewValue < 1.f)
	{
		// Do not allow Health to drop below 1
		NewValue = 1.f;
	}
}

void UFiringRangeHealthSet::PreAttributeBaseChange(const FGameplayAttribute& Attribute, float& NewValue) const
{
	Super::PreAttributeBaseChange(Attribute, NewValue);

	if (Attribute == GetHealthAttribute() && NewValue < 1.f)
	{
		// Do not allow Health to drop below 1
		NewValue = 1.f;
	}
}
