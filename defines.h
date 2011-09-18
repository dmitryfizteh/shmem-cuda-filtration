#ifndef DEFINES_H
#define DEFINES_H 

#define TEST
#define DEFINES_FILE "C://Users//dmitry//Documents//Visual Studio 2010//Projects//Buckley-Leverett//defines.ini"
#define timeX 500

#define mu1 0.001
#define mu2 0.0009
#define ro01 1000
#define ro02 1460
#define S2_gr 0.4
#define P_atm 100000
#define g_const 9.8
#define NEWTON_ITERATIONS 7

// Нитей в блоке ускорителя
#define BlockNX 16
#define BlockNY 4
#define BlockNZ 4

const double K[2]={6.64e-11,7.15e-12};
const double lambda[2]={2.7,2.0};
const double S_wr[2]={0.09,0.12};
const double m[2]={0.4,0.39};
const double P_d[2]={755,2060};

#include <stdio.h>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

struct ptr_Arrays_tag 
{
	double *x, *y, *z;
	double *P1, *P2, *S2, *ro1, *ro2, *u1x, *u1y, *u1z, *u2x, *u2y, *u2z, *Xi1, *Xi2,*roS1,*roS1_old,*roS2,*roS2_old;
	int *media;
};
typedef struct ptr_Arrays_tag ptr_Arrays;

// Структура параметров сред
struct consts_tag
{
	double K[2];
	double lambda[2];
	double S_wr[2];
	double m[2];
	double P_d[2];
	double h1, h2, h3, dt, tau, l_w, l_n, c, beta1, beta2;
	int NX, Ny, Nz;
	int source, save_plots, print_screen;
};
typedef struct consts_tag consts;

extern void time_step_function(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr, double* DevBuffer, consts def, double t, int localNx, int localNy, int rank,int size, int blocksX, int blocksY, int blocksZ);
extern void Initialize(ptr_Arrays* HostArraysPtr, ptr_Arrays* DevArraysPtr, int* j, int* localNx, int* localNy, int* size, int* rank, int* blocksX, int* blocksY, int* blocksZ, int argc, char* argv[], consts def);
extern void Finalize(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr, double* DevBuffer);
extern void memory_alloc(ptr_Arrays* HostArraysPtr, ptr_Arrays* DevArraysPtr, int localNx, int nY, consts def);
extern void host_memory_alloc(ptr_Arrays* ArraysPtr, int nX, int nY, consts def);
extern void device_memory_alloc(ptr_Arrays* ArraysPtr, double** DevBuffer, int nX, consts def);
extern void memory_free(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr);
extern void host_memory_free(ptr_Arrays HostArraysPtr);
extern void device_memory_free(ptr_Arrays DevArraysPtr, double* DevBuffer);
extern void save_data_plots(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr, double t, int size, int rank, int localNx, consts def);
extern void initial_data(ptr_Arrays HostArraysPtr, int* t, int localNx, int localNy, int rank, int size, consts def);
extern void Communication_Initialize(int argc, char* argv[], int* size, int* rank, consts def);
extern void Communication_Finalize(void);
extern void N_to_local (int* localNx, int* localNy, int size, int rank, consts def);
extern int is_active_point(int i, int localNx, int rank, int size);
extern void load_data_to_host(double* HostArrayPtr, double* DevArrayPtr, int localNx, consts def);
extern void load_data_to_device(double* HostArrayPtr, double* DevArrayPtr, int localNx, consts def);
extern void load_data_to_device_int(int* HostArrayPtr, int* DevArrayPtr, int localNx, consts def);

extern void Device_Initialize(int rank, int* blocksX, int* blocksY, int* blocksZ, int localNx, consts def);

// Служебные
extern void print_plots_top (double t, consts def);
extern void print_plots(ptr_Arrays HostArraysPtr, double t, int rank, int size, int localNx, consts def);
extern void Barrier(void);
extern void restore (ptr_Arrays HostArraysPtr, int* j, int rank, int size, int localNx, consts def);
extern void save(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr, int j, int rank, int size, int localNx, consts def);
extern void read_defines(int argc, char *argv[], consts* def);

// Unit-тесты
extern void test_correct_P1_S2(ptr_Arrays HostArraysPtr, int nX, int rank, consts def);

// Расчеты в каждой точке
extern int i_to_I(int i, int rank, int size, consts def);
extern void assign_P2_Xi1_Xi2(ptr_Arrays HostArraysPtr, int i, int j, int k, int localNx, consts def);
extern void assign_ro1_ro2(ptr_Arrays HostArraysPtr, int i, int j, int k, int localNx, consts def);
extern void assign_u(ptr_Arrays HostArraysPtr, int i, int j, int k, int localNx, consts def);
extern void assign_rS(ptr_Arrays HostArraysPtr, double t, int i, int j, int k, int localNx, consts def);
extern void assign_rS_nr(ptr_Arrays HostArraysPtr, double t, int i, int j, int k, int localNx, consts def);
extern void Newton(ptr_Arrays HostArraysPtr, int i, int j, int k, int localNx, consts def);
extern void Border_S2(ptr_Arrays HostArraysPtr, int i, int j, int k, int localNx, int rank, int size, consts def);
extern void Border_P1(ptr_Arrays HostArraysPtr, int i, int j, int k, int localNx, consts def);

extern void ro_P2_Xi_calculation(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr, consts def, int localNx, int rank, int size, int blocksX, int blocksY, int blocksZ);
extern void P2_ro_Xi_exchange(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr, double* HostBuffer, double* DevBuffer, int localNx, int blocksY, int blocksZ, int rank, int size, consts def);
extern void u_calculation(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr, int localNx, int rank, int size, int blocksX, int blocksY, int blocksZ, consts def);
extern void u_exchange(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr, double* HostBuffer, double* DevBuffer, int localNx, int blocksY, int blocksZ, int rank, int size, consts def);
extern void roS_calculation(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr, consts def, double t, int localNx, int rank, int size, int blocksX, int blocksY, int blocksZ);
extern void P1_S2_calculation(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr, consts def, int localNx, int rank, int size, int blocksX, int blocksY, int blocksZ);
extern void boundary_conditions(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr, int localNx, int rank, int size, int blocksX, int blocksY, int blocksZ, consts def);
extern void P1_S2_exchange(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr, double* HostBuffer, double* DevBuffer, int localNx, int blocksY, int blocksZ, int rank, int size, consts def);

extern void load_exchange_data(double* HostArrayPtr, double* DevArrayPtr, double* HostBuffer, double* DevBuffer, int localNx, int blocksY, int blocksZ, int rank, int size, consts def);
extern void save_exchange_data(double* HostArrayPtr, double* DevArrayPtr, double* HostBuffer, double* DevBuffer, int localNx, int blocksY, int blocksZ, int rank, int size, consts def);

#endif