################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../main.c \
../ridge.c \
../util.c 

OBJS += \
./main.o \
./ridge.o \
./util.o 

C_DEPS += \
./main.d \
./ridge.d \
./util.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -I/Users/jiangp4/anaconda3/lib/python3.7/site-packages/numpy/core/include/numpy -I/Users/jiangp4/anaconda3/include/python3.7m -O3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


