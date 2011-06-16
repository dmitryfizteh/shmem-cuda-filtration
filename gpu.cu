#include "defines.h"
#include <cuda.h>

__constant__ consts gpu_def [1];

// Проверка ошибок GPU
void checkErrors(char *label) 
{
#ifdef TEST
	cudaError_t err;

	err = cudaThreadSynchronize();
	if (err != cudaSuccess)
	{
		char *e = (char*) cudaGetErrorString(err);
		printf("CUDA Error: %s (at %s)\n", e, label);
	}

	err=cudaGetLastError();
	if (err != cudaSuccess)
	{
		char *e = (char*) cudaGetErrorString(err);
		printf("CUDA Error: %s (at %s)\n", e, label);
		fflush(stdout);
	}
#endif
}

// Является ли точка активной (т.е. не предназначенной только для обмена на границах)
__device__ int device_is_active_point(int i, int localNx, int rank, int size)
{
	if((rank!=0 && i==0) || (rank!=size-1 && i==localNx-1))
		return 0;
	else
		return 1;
}

// Преобразование локальных координат процессора к глобальным
// Каждый процессор содержит дополнительную точку в массиве для
// обмена данными, если имеет соседа 
// (если 2 соседа с обеих сторон,то +2 точки). 
// Глобальные границы хранятся как обычные точки (отсюда и условие на rank==0)
__device__ int device_i_to_I(int i, int rank, int size, consts def)
{
	int I;
	if (rank <= ((*gpu_def).NX)%size)
	{
		if(rank==0)
			I=i;
		else
			I=(((*gpu_def).NX)/size+1)*rank+i-1;
	}
	else
		I=(((*gpu_def).NX)/size+1)*rank-(rank-((*gpu_def).NX)%size)+i-1;
	return I;
}

// Расчет плотностей, давления NAPL P2 и Xi в каждой точке сетки (независимо от остальных точек)
__global__ void assign_P2_Xi1_Xi2_ro1_ro2_kernel(ptr_Arrays DevArraysPtr, int localNx, int rank, int size) 
{
	int blockIdxz=blockIdx.y / BlockNY;
	int blockIdxy=blockIdx.y % BlockNY;
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	int k=threadIdx.z+blockIdxz*blockDim.z;
	int j=threadIdx.y+blockIdxy*blockDim.y;

	if ((i<localNx) && (j<((*gpu_def).Ny)) && (k<((*gpu_def).Nz)) && (device_is_active_point(i, localNx, rank, size)==1))
	{
		int media = DevArraysPtr.media[i+j*localNx+k*localNx*((*gpu_def).Ny)];
		double S2 = DevArraysPtr.S2[i+j*localNx+k*localNx*((*gpu_def).Ny)];
		double P1 = DevArraysPtr.P1[i+j*localNx+k*localNx*((*gpu_def).Ny)];

		double S_e = (1 - S2 - (*gpu_def).S_wr[media]) / (1 - (*gpu_def).S_wr[media]);
		double k1 = pow(S_e, (2. + 3. * (*gpu_def).lambda[media]) / (*gpu_def).lambda[media]);
		double k2 = (1. - S_e) * (1. - S_e) * (1 - pow(S_e, (2. + (*gpu_def).lambda[media]) / (*gpu_def).lambda[media]));
		double P_k = (*gpu_def).P_d[media] * pow((1 - S2 - (*gpu_def).S_wr[media]) / (1 - (*gpu_def).S_wr[media]), -1 / (*gpu_def).lambda[media]);

		DevArraysPtr.P2[i+j*localNx+k*localNx*((*gpu_def).Ny)] = P1 + P_k;
		DevArraysPtr.Xi1[i+j*localNx+k*localNx*((*gpu_def).Ny)] = -1 * (*gpu_def).K[media] * k1 / mu1;
		DevArraysPtr.Xi2[i+j*localNx+k*localNx*((*gpu_def).Ny)] = -1 * (*gpu_def).K[media] * k2 / mu2;
		DevArraysPtr.ro1[i+j*localNx+k*localNx*((*gpu_def).Ny)] = ro01 * (1 + ((*gpu_def).beta1) * (P1 - P_atm));
		DevArraysPtr.ro2[i+j*localNx+k*localNx*((*gpu_def).Ny)] = ro02 * (1 + ((*gpu_def).beta2) * (P1 + P_k - P_atm));
	}
}

// Расчет плотностей, давления NAPL P2 и Xi во всех точках сетки
void ro_P2_Xi_calculation(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr, consts def, int localNx, int rank, int size, int blocksX, int blocksY, int blocksZ)
{
	assign_P2_Xi1_Xi2_ro1_ro2_kernel<<<dim3(blocksX,blocksY*blocksZ), dim3(BlockNX,BlockNY,BlockNZ)>>>(DevArraysPtr,localNx,rank,size); 
	checkErrors("assign P2, Xi,ro");
}

// Расчет скорости в каждой точке сетки
__global__ void assign_u_kernel(ptr_Arrays DevArraysPtr, int localNx, int rank, int size) 
{
	int blockIdxz=blockIdx.y / BlockNY;
	int blockIdxy=blockIdx.y % BlockNY;
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	int k=threadIdx.z+blockIdxz*blockDim.z;
	int j=threadIdx.y+blockIdxy*blockDim.y;

	if ((i<localNx) && (j<((*gpu_def).Ny)) && (k<((*gpu_def).Nz)) && (device_is_active_point(i, localNx, rank, size)==1))
	{
		double Xi1 = DevArraysPtr.Xi1[i+j*localNx+k*localNx*((*gpu_def).Ny)];
		double Xi2 = DevArraysPtr.Xi2[i+j*localNx+k*localNx*((*gpu_def).Ny)];
		double P1 = DevArraysPtr.P1[i+j*localNx+k*localNx*((*gpu_def).Ny)];
		double P2 = DevArraysPtr.P2[i+j*localNx+k*localNx*((*gpu_def).Ny)];

		if (((*gpu_def).NX)>2)
		{
			if (i == 0)
			{
				DevArraysPtr.u1x[i+j*localNx+k*localNx*((*gpu_def).Ny)] = Xi1 * (DevArraysPtr.P1[i+1+j*localNx+k*localNx*((*gpu_def).Ny)] - P1) / ((*gpu_def).h1);
				DevArraysPtr.u2x[i+j*localNx+k*localNx*((*gpu_def).Ny)] = Xi2 * (DevArraysPtr.P2[i+1+j*localNx+k*localNx*((*gpu_def).Ny)] - P2) / ((*gpu_def).h1);
			}
			if (i == localNx - 1)
			{
				DevArraysPtr.u1x[i+j*localNx+k*localNx*((*gpu_def).Ny)] = Xi1 * (P1 - DevArraysPtr.P1[i-1+j*localNx+k*localNx*((*gpu_def).Ny)]) / ((*gpu_def).h1);
				DevArraysPtr.u2x[i+j*localNx+k*localNx*((*gpu_def).Ny)] = Xi2 * (P2 - DevArraysPtr.P2[i-1+j*localNx+k*localNx*((*gpu_def).Ny)]) / ((*gpu_def).h1);
			}
			if ((i != 0) && (i != localNx - 1))
			{
				DevArraysPtr.u1x[i+j*localNx+k*localNx*((*gpu_def).Ny)] = Xi1 * ((DevArraysPtr.P1[i+1+j*localNx+k*localNx*((*gpu_def).Ny)] - DevArraysPtr.P1[i-1+j*localNx+k*localNx*((*gpu_def).Ny)]) / (2 * ((*gpu_def).h1)));
				DevArraysPtr.u2x[i+j*localNx+k*localNx*((*gpu_def).Ny)] = Xi2 * ((DevArraysPtr.P2[i+1+j*localNx+k*localNx*((*gpu_def).Ny)] - DevArraysPtr.P2[i-1+j*localNx+k*localNx*((*gpu_def).Ny)]) / (2 * ((*gpu_def).h1)));
			}
		}
		else
		{
			DevArraysPtr.u1x[i+j*localNx+k*localNx*((*gpu_def).Ny)] = 0;
			DevArraysPtr.u2x[i+j*localNx+k*localNx*((*gpu_def).Ny)] = 0;
		}
	
		if (((*gpu_def).Ny)>2)
		{
			if (j == 0)
			{
				DevArraysPtr.u1y[i+j*localNx+k*localNx*((*gpu_def).Ny)] = Xi1 * ((DevArraysPtr.P1[i+(j+1)*localNx+k*localNx*((*gpu_def).Ny)] - P1) / ((*gpu_def).h2) - DevArraysPtr.ro1[i+j*localNx+k*localNx*((*gpu_def).Ny)] * g_const);
				DevArraysPtr.u2y[i+j*localNx+k*localNx*((*gpu_def).Ny)] = Xi2 * ((DevArraysPtr.P2[i+(j+1)*localNx+k*localNx*((*gpu_def).Ny)] - P2) / ((*gpu_def).h2) - DevArraysPtr.ro2[i+j*localNx+k*localNx*((*gpu_def).Ny)] * g_const);
			}

			if (j == ((*gpu_def).Ny) - 1)
			{
				DevArraysPtr.u1y[i+j*localNx+k*localNx*((*gpu_def).Ny)] = Xi1 * ((P1 - DevArraysPtr.P1[i+(j-1)*localNx+k*localNx*((*gpu_def).Ny)]) / ((*gpu_def).h2) - DevArraysPtr.ro1[i+j*localNx+k*localNx*((*gpu_def).Ny)] * g_const);
				DevArraysPtr.u2y[i+j*localNx+k*localNx*((*gpu_def).Ny)] = Xi2 * ((P2 - DevArraysPtr.P2[i+(j-1)*localNx+k*localNx*((*gpu_def).Ny)]) / ((*gpu_def).h2) - DevArraysPtr.ro2[i+j*localNx+k*localNx*((*gpu_def).Ny)] * g_const);
			}
			if ((j != 0) && (j != ((*gpu_def).Ny) - 1))
			{
				DevArraysPtr.u1y[i+j*localNx+k*localNx*((*gpu_def).Ny)] = Xi1 * ((DevArraysPtr.P1[i+(j+1)*localNx+k*localNx*((*gpu_def).Ny)] - DevArraysPtr.P1[i+(j-1)*localNx+k*localNx*((*gpu_def).Ny)]) / (2 * ((*gpu_def).h2)) - DevArraysPtr.ro1[i+j*localNx+k*localNx*((*gpu_def).Ny)] * g_const);
				DevArraysPtr.u2y[i+j*localNx+k*localNx*((*gpu_def).Ny)] = Xi2 * ((DevArraysPtr.P2[i+(j+1)*localNx+k*localNx*((*gpu_def).Ny)] - DevArraysPtr.P2[i+(j-1)*localNx+k*localNx*((*gpu_def).Ny)]) / (2 * ((*gpu_def).h2)) - DevArraysPtr.ro2[i+j*localNx+k*localNx*((*gpu_def).Ny)] * g_const);
			}
		}
		else
		{
			DevArraysPtr.u1y[i+j*localNx+k*localNx*((*gpu_def).Ny)] = 0;
			DevArraysPtr.u2y[i+j*localNx+k*localNx*((*gpu_def).Ny)] = 0;
		}

		if (((*gpu_def).Nz)>2)
		{
			if (k == 0)
			{
				DevArraysPtr.u1z[i+j*localNx+k*localNx*((*gpu_def).Ny)] = Xi1 * (DevArraysPtr.P1[i+j*localNx+(k+1)*localNx*((*gpu_def).Ny)] - P1) / ((*gpu_def).h3);
				DevArraysPtr.u2z[i+j*localNx+k*localNx*((*gpu_def).Ny)] = Xi2 * (DevArraysPtr.P2[i+j*localNx+(k+1)*localNx*((*gpu_def).Ny)] - P2) / ((*gpu_def).h3);
			}
			if (k == ((*gpu_def).Nz) - 1)
			{
				DevArraysPtr.u1z[i+j*localNx+k*localNx*((*gpu_def).Ny)] = Xi1 * (P1 - DevArraysPtr.P1[i+j*localNx+(k-1)*localNx*((*gpu_def).Ny)]) / ((*gpu_def).h3);
				DevArraysPtr.u2z[i+j*localNx+k*localNx*((*gpu_def).Ny)] = Xi2 * (P2 - DevArraysPtr.P2[i+j*localNx+(k-1)*localNx*((*gpu_def).Ny)]) / ((*gpu_def).h3);
			}
			if ((k != 0) && (i != ((*gpu_def).Nz) - 1))
			{
				DevArraysPtr.u1z[i+j*localNx+k*localNx*((*gpu_def).Ny)] = Xi1 * ((DevArraysPtr.P1[i+j*localNx+(k+1)*localNx*((*gpu_def).Ny)] - DevArraysPtr.P1[i+j*localNx+(k-1)*localNx*((*gpu_def).Ny)]) / (2 * ((*gpu_def).h3)));
				DevArraysPtr.u2z[i+j*localNx+k*localNx*((*gpu_def).Ny)] = Xi2 * ((DevArraysPtr.P2[i+j*localNx+(k+1)*localNx*((*gpu_def).Ny)] - DevArraysPtr.P2[i+j*localNx+(k-1)*localNx*((*gpu_def).Ny)]) / (2 * ((*gpu_def).h3)));
			}
		}
		else
		{
			DevArraysPtr.u1z[i+j*localNx+k*localNx*((*gpu_def).Ny)] = 0;
			DevArraysPtr.u2z[i+j*localNx+k*localNx*((*gpu_def).Ny)] = 0;
		}
	}
}

// Расчет скоростей во всех точках сетки
void u_calculation(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr, int localNx, int rank, int size, int blocksX, int blocksY, int blocksZ, consts def)
{
	assign_u_kernel<<<dim3(blocksX,blocksY*blocksZ), dim3(BlockNX,BlockNY,BlockNZ)>>>(DevArraysPtr,localNx,rank,size); 
	checkErrors("assign u");
}
// Расчет ro*S в каждой точке сетки методом направленных разностей
__global__ void assign_rS_kernel_nr(ptr_Arrays DevArraysPtr, int localNx, double t)
{
	int blockIdxz=blockIdx.y / BlockNY;
	int blockIdxy=blockIdx.y % BlockNY;
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	int k=threadIdx.z+blockIdxz*blockDim.z;
	int j=threadIdx.y+blockIdxy*blockDim.y;
	
	if ((i<localNx-1) && (j<(*gpu_def).Ny-1) && (k<(*gpu_def).Nz) && (i!=0) && (j!=0) && (((k!=0) && (k!=(*gpu_def).Nz-1)) || ((*gpu_def).Nz<2)))
	{
		int media = DevArraysPtr.media[i+j*localNx+k*localNx*((*gpu_def).Ny)];

		double S2 = DevArraysPtr.S2[i+j*localNx+k*localNx*((*gpu_def).Ny)];
		double roS1 = DevArraysPtr.ro1[i+j*localNx+k*localNx*((*gpu_def).Ny)] * (1 - S2);
		double roS2 = DevArraysPtr.ro2[i+j*localNx+k*localNx*((*gpu_def).Ny)] * S2;
 		double P1 = DevArraysPtr.P1[i+j*localNx+k*localNx*((*gpu_def).Ny)];
		double P2 = DevArraysPtr.P2[i+j*localNx+k*localNx*((*gpu_def).Ny)];

		double x1, x2, y1, y2, z1, z2, f1, f2, f3, g1, g2, g3;

		if (((*gpu_def).Nz)<2)
		{
			f3=0;
			g3=0;
		}
		else
		{
			z2 = -(DevArraysPtr.P1[i+j*localNx+(k+1)*localNx*((*gpu_def).Ny)] - P1)/(*gpu_def).h3;
			z1 = -(P1 - DevArraysPtr.P1[i+j*localNx+(k-1)*localNx*((*gpu_def).Ny)])/(*gpu_def).h3;

			f3 = (((z2 + abs(z2))/2.0 - (z1 - abs(z1))/2.0)*(-1) * DevArraysPtr.Xi1[i+j*localNx+k*localNx*((*gpu_def).Ny)] * DevArraysPtr.ro1[i+j*localNx+k*localNx*((*gpu_def).Ny)] -
                      (z1 + abs(z1))/2.0*(-1)* DevArraysPtr.Xi1[i+j*localNx+(k-1)*localNx*((*gpu_def).Ny)] * DevArraysPtr.ro1[i+j*localNx+(k-1)*localNx*((*gpu_def).Ny)] +
                      (z2 - abs(z2))/2.0*(-1)* DevArraysPtr.Xi1[i+j*localNx+(k+1)*localNx*((*gpu_def).Ny)] * DevArraysPtr.ro1[i+j*localNx+(k+1)*localNx*((*gpu_def).Ny)])/(*gpu_def).h3;

			z2 = -(DevArraysPtr.P2[i+j*localNx+(k+1)*localNx*((*gpu_def).Ny)] - P2)/(*gpu_def).h3;
			z1 = -(P2 - DevArraysPtr.P2[i+j*localNx+(k-1)*localNx*((*gpu_def).Ny)])/(*gpu_def).h3;

			g3 = (((z2 + abs(z2))/2.0 - (z1 - abs(z1))/2.0)*(-1) * DevArraysPtr.Xi2[i+j*localNx+k*localNx*((*gpu_def).Ny)] * DevArraysPtr.ro2[i+j*localNx+k*localNx*((*gpu_def).Ny)] -
                      (z1 + abs(z1))/2.0*(-1)* DevArraysPtr.Xi2[i+j*localNx+(k-1)*localNx*((*gpu_def).Ny)] * DevArraysPtr.ro2[i+j*localNx+(k-1)*localNx*((*gpu_def).Ny)] +
                      (z2 - abs(z2))/2.0*(-1)* DevArraysPtr.Xi2[i+j*localNx+(k+1)*localNx*((*gpu_def).Ny)] * DevArraysPtr.ro2[i+j*localNx+(k+1)*localNx*((*gpu_def).Ny)])/(*gpu_def).h3;
		}

		x2 = -(DevArraysPtr.P1[i+1+j*localNx+k*localNx*((*gpu_def).Ny)] - P1)/(*gpu_def).h1;
        x1 = -(P1 - DevArraysPtr.P1[i-1+j*localNx+k*localNx*((*gpu_def).Ny)])/(*gpu_def).h1;

        y2 = -(DevArraysPtr.P1[i+(j+1)*localNx+k*localNx*((*gpu_def).Ny)] - P1)/(*gpu_def).h2 + DevArraysPtr.ro1[i+j*localNx+k*localNx*((*gpu_def).Ny)] * g_const;
        y1 = -(P1 - DevArraysPtr.P1[i+(j-1)*localNx+k*localNx*((*gpu_def).Ny)])/(*gpu_def).h2 + DevArraysPtr.ro1[i+j*localNx+k*localNx*((*gpu_def).Ny)] * g_const;

        f1 = (((x2 + abs(x2))/2.0 - (x1 - abs(x1))/2.0)*(-1) * DevArraysPtr.Xi1[i+j*localNx+k*localNx*((*gpu_def).Ny)] * DevArraysPtr.ro1[i+j*localNx+k*localNx*((*gpu_def).Ny)] -
                (x1 + abs(x1))/2.0*(-1)* DevArraysPtr.Xi1[i-1+j*localNx+k*localNx*((*gpu_def).Ny)] * DevArraysPtr.ro1[i-1+j*localNx+k*localNx*((*gpu_def).Ny)] +
                (x2 - abs(x2))/2.0*(-1)* DevArraysPtr.Xi1[i+1+j*localNx+k*localNx*((*gpu_def).Ny)] * DevArraysPtr.ro1[i+1+j*localNx+k*localNx*((*gpu_def).Ny)])/(*gpu_def).h1;

        f2 = (((y2 + abs(y2))/2.0 - (y1 - abs(y1))/2.0)*(-1)* DevArraysPtr.Xi1[i+j*localNx+k*localNx*((*gpu_def).Ny)] * DevArraysPtr.ro1[i+j*localNx+k*localNx*((*gpu_def).Ny)] -
                (y1 + abs(y1))/2.0*(-1)* DevArraysPtr.Xi1[i+(j-1)*localNx+k*localNx*((*gpu_def).Ny)] * DevArraysPtr.ro1[i+(j-1)*localNx+k*localNx*((*gpu_def).Ny)] +
                (y2 - abs(y2))/2.0*(-1)* DevArraysPtr.Xi1[i+(j+1)*localNx+k*localNx*((*gpu_def).Ny)] * DevArraysPtr.ro1[i+(j+1)*localNx+k*localNx*((*gpu_def).Ny)])/(*gpu_def).h2;


        x2 = -(DevArraysPtr.P2[i+1+j*localNx+k*localNx*((*gpu_def).Ny)] - P2)/(*gpu_def).h1;
        x1 = -(P2 - DevArraysPtr.P2[i-1+j*localNx+k*localNx*((*gpu_def).Ny)])/(*gpu_def).h1;

        y2 = -(DevArraysPtr.P2[i+(j+1)*localNx+k*localNx*((*gpu_def).Ny)] - P2)/(*gpu_def).h2 + DevArraysPtr.ro2[i+j*localNx+k*localNx*((*gpu_def).Ny)] * g_const;
        y1 = -(P2 - DevArraysPtr.P2[i+(j-1)*localNx+k*localNx*((*gpu_def).Ny)])/(*gpu_def).h2 + DevArraysPtr.ro2[i+j*localNx+k*localNx*((*gpu_def).Ny)] * g_const;

        g1 = (((x2 + abs(x2))/2.0 - (x1 - abs(x1))/2.0)*(-1) * DevArraysPtr.Xi2[i+j*localNx+k*localNx*((*gpu_def).Ny)] * DevArraysPtr.ro2[i+j*localNx+k*localNx*((*gpu_def).Ny)] -
                (x1 + abs(x1))/2.0*(-1)* DevArraysPtr.Xi2[i-1+j*localNx+k*localNx*((*gpu_def).Ny)] * DevArraysPtr.ro2[i-1+j*localNx+k*localNx*((*gpu_def).Ny)] +
                (x2 - abs(x2))/2.0*(-1)* DevArraysPtr.Xi2[i+1+j*localNx+k*localNx*((*gpu_def).Ny)] * DevArraysPtr.ro2[i+1+j*localNx+k*localNx*((*gpu_def).Ny)])/(*gpu_def).h1;

        g2 = (((y2 + abs(y2))/2.0 - (y1 - abs(y1))/2.0)*(-1)* DevArraysPtr.Xi2[i+j*localNx+k*localNx*((*gpu_def).Ny)] * DevArraysPtr.ro2[i+j*localNx+k*localNx*((*gpu_def).Ny)] -
                (y1 + abs(y1))/2.0*(-1)* DevArraysPtr.Xi2[i+(j-1)*localNx+k*localNx*((*gpu_def).Ny)] * DevArraysPtr.ro2[i+(j-1)*localNx+k*localNx*((*gpu_def).Ny)] +
                (y2 - abs(y2))/2.0*(-1)* DevArraysPtr.Xi2[i+(j+1)*localNx+k*localNx*((*gpu_def).Ny)] * DevArraysPtr.ro2[i+(j+1)*localNx+k*localNx*((*gpu_def).Ny)])/(*gpu_def).h2;

		DevArraysPtr.roS1_old[i+j*localNx+k*localNx*((*gpu_def).Ny)] = roS1;
		DevArraysPtr.roS2_old[i+j*localNx+k*localNx*((*gpu_def).Ny)] = roS2;
		DevArraysPtr.roS1[i+j*localNx+k*localNx*((*gpu_def).Ny)] = roS1 - ((*gpu_def).dt/(*gpu_def).m[media])*(f1 + f2 + f3);
		DevArraysPtr.roS2[i+j*localNx+k*localNx*((*gpu_def).Ny)] = roS2 - ((*gpu_def).dt/(*gpu_def).m[media])*(g1 + g2 + g3);
	}
}

// Расчет ro*S в каждой точке сетки
__global__ void assign_rS_kernel(ptr_Arrays DevArraysPtr, int localNx, double t) 
{
	int blockIdxz=blockIdx.y / BlockNY;
	int blockIdxy=blockIdx.y % BlockNY;
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	int k=threadIdx.z+blockIdxz*blockDim.z;
	int j=threadIdx.y+blockIdxy*blockDim.y;

	if ((i<localNx-1) && (j<(*gpu_def).Ny-1) && (k<(*gpu_def).Nz) && (i!=0) && (j!=0) && (((k!=0) && (k!=(*gpu_def).Nz-1)) || ((*gpu_def).Nz<2)))
	{
		int local=i+j*localNx+k*localNx*((*gpu_def).Ny);
		int media = DevArraysPtr.media[local];
		double S2 = DevArraysPtr.S2[local];
		double roS1 = DevArraysPtr.ro1[local] * (1 - S2);
		double roS2 = DevArraysPtr.ro2[local] * S2;

		double divgrad1, divgrad2, Tx1, Ty1, Tx2, Ty2, Tz1, Tz2, A1=0, A2=0;

		if (((*gpu_def).Nz)<2)
		{
			divgrad1=0;
			divgrad2=0;
			Tz1=0;
			Tz2=0;
		}
		else
		{
			divgrad1 = ((*gpu_def).m[media] * ((*gpu_def).l_w) * ((*gpu_def).c) / 2.) * (DevArraysPtr.ro1[i+j*localNx+(k+1)*localNx*((*gpu_def).Ny)] * (1. - DevArraysPtr.S2[i+j*localNx+(k+1)*localNx*((*gpu_def).Ny)]) - 2 * DevArraysPtr.ro1[local] * (1. - S2) + DevArraysPtr.ro1[i+j*localNx+(k-1)*localNx*((*gpu_def).Ny)] * (1. - DevArraysPtr.S2[i+j*localNx+(k-1)*localNx*((*gpu_def).Ny)])) / (((*gpu_def).h3) * ((*gpu_def).h3));
			divgrad2 = ((*gpu_def).m[media] * ((*gpu_def).l_n) * ((*gpu_def).c) / 2.) * (DevArraysPtr.ro2[i+j*localNx+(k+1)*localNx*((*gpu_def).Ny)] * DevArraysPtr.S2[i+j*localNx+(k+1)*localNx*((*gpu_def).Ny)] - 2 * DevArraysPtr.ro2[local] * S2 + DevArraysPtr.ro2[i+j*localNx+(k-1)*localNx*((*gpu_def).Ny)] * (DevArraysPtr.S2[i+j*localNx+(k-1)*localNx*((*gpu_def).Ny)])) / (((*gpu_def).h3) * ((*gpu_def).h3));
			Tz1 = (DevArraysPtr.ro1[i+1+j*localNx+(k+1)*localNx*((*gpu_def).Ny)] * DevArraysPtr.u1x[i+1+j*localNx+(k+1)*localNx*((*gpu_def).Ny)] - DevArraysPtr.ro1[i+j*localNx+(k-1)*localNx*((*gpu_def).Ny)] * DevArraysPtr.u1x[i+j*localNx+(k-1)*localNx*((*gpu_def).Ny)]) / (2. * ((*gpu_def).h3));
			Tz2 = (DevArraysPtr.ro2[i+j*localNx+(k+1)*localNx*((*gpu_def).Ny)] * DevArraysPtr.u2y[i+j*localNx+(k+1)*localNx*((*gpu_def).Ny)] - DevArraysPtr.ro2[i+j*localNx+(k-1)*localNx*((*gpu_def).Ny)] * DevArraysPtr.u2y[i+j*localNx+(k-1)*localNx*((*gpu_def).Ny)]) / (2. * ((*gpu_def).h3));
		}

		divgrad1 += ((*gpu_def).m[media] * ((*gpu_def).l_w) * ((*gpu_def).c) / 2.) *
		((DevArraysPtr.ro1[i+1+j*localNx+k*localNx*((*gpu_def).Ny)] * (1 - DevArraysPtr.S2[i+1+j*localNx+k*localNx*((*gpu_def).Ny)]) - 2 * DevArraysPtr.ro1[i+j*localNx+k*localNx*((*gpu_def).Ny)] * (1 - S2) + DevArraysPtr.ro1[i-1+j*localNx+k*localNx*((*gpu_def).Ny)] * (1 - DevArraysPtr.S2[i-1+j*localNx+k*localNx*((*gpu_def).Ny)])) / (((*gpu_def).h1) * ((*gpu_def).h1)) +
		(DevArraysPtr.ro1[i+(j+1)*localNx+k*localNx*((*gpu_def).Ny)] * (1 - DevArraysPtr.S2[i+(j+1)*localNx+k*localNx*((*gpu_def).Ny)]) - 2 * DevArraysPtr.ro1[i+j*localNx+k*localNx*((*gpu_def).Ny)] * (1 - S2) + DevArraysPtr.ro1[i+(j-1)*localNx+k*localNx*((*gpu_def).Ny)] * (1 - DevArraysPtr.S2[i+(j-1)*localNx+k*localNx*((*gpu_def).Ny)])) / (((*gpu_def).h2) * ((*gpu_def).h2)));

		divgrad2 += ((*gpu_def).m[media] * ((*gpu_def).l_n) * ((*gpu_def).c) / 2.) *
		((DevArraysPtr.ro2[i+1+j*localNx+k*localNx*((*gpu_def).Ny)] * DevArraysPtr.S2[i+1+j*localNx+k*localNx*((*gpu_def).Ny)] - 2 * DevArraysPtr.ro2[i+j*localNx+k*localNx*((*gpu_def).Ny)] * S2 + DevArraysPtr.ro2[i-1+j*localNx+k*localNx*((*gpu_def).Ny)] * DevArraysPtr.S2[i-1+j*localNx+k*localNx*((*gpu_def).Ny)]) / (((*gpu_def).h1) * ((*gpu_def).h1)) +
		(DevArraysPtr.ro2[i+(j+1)*localNx+k*localNx*((*gpu_def).Ny)] * DevArraysPtr.S2[i+(j+1)*localNx+k*localNx*((*gpu_def).Ny)] - 2 * DevArraysPtr.ro2[i+j*localNx+k*localNx*((*gpu_def).Ny)] * S2 + DevArraysPtr.ro2[i+(j-1)*localNx+k*localNx*((*gpu_def).Ny)] * DevArraysPtr.S2[i+(j-1)*localNx+k*localNx*((*gpu_def).Ny)]) / (((*gpu_def).h2) * ((*gpu_def).h2)));

		Tx1 = (DevArraysPtr.ro1[i+1+j*localNx+k*localNx*((*gpu_def).Ny)] * DevArraysPtr.u1x[i+1+j*localNx+k*localNx*((*gpu_def).Ny)] - DevArraysPtr.ro1[i-1+j*localNx+k*localNx*((*gpu_def).Ny)] * DevArraysPtr.u1x[i-1+j*localNx+k*localNx*((*gpu_def).Ny)]) / (2 * ((*gpu_def).h1));
		Ty1 = (DevArraysPtr.ro1[i+(j+1)*localNx+k*localNx*((*gpu_def).Ny)] * DevArraysPtr.u1y[i+(j+1)*localNx+k*localNx*((*gpu_def).Ny)] - DevArraysPtr.ro1[i+(j-1)*localNx+k*localNx*((*gpu_def).Ny)] * DevArraysPtr.u1y[i+(j-1)*localNx+k*localNx*((*gpu_def).Ny)]) / (2 * ((*gpu_def).h2));
		Tx2 = (DevArraysPtr.ro2[i+1+j*localNx+k*localNx*((*gpu_def).Ny)] * DevArraysPtr.u2x[i+1+j*localNx+k*localNx*((*gpu_def).Ny)] - DevArraysPtr.ro2[i-1+j*localNx+k*localNx*((*gpu_def).Ny)] * DevArraysPtr.u2x[i-1+j*localNx+k*localNx*((*gpu_def).Ny)]) / (2 * ((*gpu_def).h1));
		Ty2 = (DevArraysPtr.ro2[i+(j+1)*localNx+k*localNx*((*gpu_def).Ny)] * DevArraysPtr.u2y[i+(j+1)*localNx+k*localNx*((*gpu_def).Ny)] - DevArraysPtr.ro2[i+(j-1)*localNx+k*localNx*((*gpu_def).Ny)] * DevArraysPtr.u2y[i+(j-1)*localNx+k*localNx*((*gpu_def).Ny)]) / (2 * ((*gpu_def).h2));

		if (t < 2 * ((*gpu_def).dt))
		{
			A1 = roS1 + (((*gpu_def).dt) / (*gpu_def).m[media]) * (divgrad1 - Tx1 - Ty1 - Tz1);
			A2 = roS2 + (((*gpu_def).dt) / (*gpu_def).m[media]) * (divgrad2 - Tx2 - Ty2 - Tz2);
		}
		else
		{
			A1 = (2 * ((*gpu_def).dt) * ((*gpu_def).dt)) / ((*gpu_def).m[media] * (((*gpu_def).dt) + 2 * ((*gpu_def).tau))) * (divgrad1 - Tx1 - Ty1 - Tz1 + (2 * roS1 * (*gpu_def).m[media] * ((*gpu_def).tau)) / (((*gpu_def).dt) * ((*gpu_def).dt)) + DevArraysPtr.roS1_old[i+j*localNx+k*localNx*((*gpu_def).Ny)] * (*gpu_def).m[media] * (((*gpu_def).dt) - 2 * ((*gpu_def).tau)) / (2 * ((*gpu_def).dt) * ((*gpu_def).dt)));
			A2 = (2 * ((*gpu_def).dt) * ((*gpu_def).dt)) / ((*gpu_def).m[media] * (((*gpu_def).dt) + 2 * ((*gpu_def).tau))) * (divgrad2 - Tx2 - Ty2 - Tz2 + (2 * roS2 * (*gpu_def).m[media] * ((*gpu_def).tau)) / (((*gpu_def).dt) * ((*gpu_def).dt)) + DevArraysPtr.roS2_old[i+j*localNx+k*localNx*((*gpu_def).Ny)] * (*gpu_def).m[media] * (((*gpu_def).dt) - 2 * ((*gpu_def).tau)) / (2 * ((*gpu_def).dt) * ((*gpu_def).dt)));
		}

		DevArraysPtr.roS1_old[local] = roS1;
		DevArraysPtr.roS2_old[local] = roS2;
		DevArraysPtr.roS1[local] = A1;
		DevArraysPtr.roS2[local] = A2;
	}
}

// Расчет ro*S во всех точках сетки
void roS_calculation(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr, consts def, double t, int localNx, int rank, int size, int blocksX, int blocksY, int blocksZ)
{
	assign_rS_kernel_nr<<<dim3(blocksX,blocksY*blocksZ), dim3(BlockNX,BlockNY,BlockNZ)>>>(DevArraysPtr,localNx,t); 
	checkErrors("assign roS");
}

// Метод Ньютона для каждой точки сетки (независимо от остальных точек)
__global__ void Newton_method_kernel(ptr_Arrays DevArraysPtr, int localNx) 
{
	int blockIdxz=blockIdx.y / BlockNY;
	int blockIdxy=blockIdx.y % BlockNY;
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	int k=threadIdx.z+blockIdxz*blockDim.z;
	int j=threadIdx.y+blockIdxy*blockDim.y;

	if ((i<localNx-1) && (j<(*gpu_def).Ny-1) && (k<(*gpu_def).Nz) && (i!=0) && (j!=0) && (((k!=0) && (k!=(*gpu_def).Nz-1)) || ((*gpu_def).Nz<2)))
    {
		int media = DevArraysPtr.media[i+j*localNx+k*localNx*((*gpu_def).Ny)];
        double S_e, P_k, AAA, F1, F2, PkS, F1P, F2P, F1S, F2S, det;
		double S2=DevArraysPtr.S2[i+j*localNx+k*localNx*((*gpu_def).Ny)];
		double P1=DevArraysPtr.P1[i+j*localNx+k*localNx*((*gpu_def).Ny)];

        S_e = (1 - S2 - (*gpu_def).S_wr[media]) / (1 - (*gpu_def).S_wr[media]);
        P_k = (*gpu_def).P_d[media] * pow(S_e, -1 / (*gpu_def).lambda[media]);
        AAA = pow(S_e, ((-1 / (*gpu_def).lambda[media]) - 1));
        F1 = ro01 * (1 + ((*gpu_def).beta1) * (P1 - P_atm)) * (1 - S2) - DevArraysPtr.roS1[i+j*localNx+k*localNx*((*gpu_def).Ny)];
        F2 = ro02 * (1 + ((*gpu_def).beta2) * (P1 + P_k - P_atm)) * S2 - DevArraysPtr.roS2[i+j*localNx+k*localNx*((*gpu_def).Ny)];

        PkS = AAA * (*gpu_def).P_d[media] / ((*gpu_def).lambda[media] * (1 - (*gpu_def).S_wr[media]));
        F1P = ro01 * ((*gpu_def).beta1) * (1 - S2);
        F2P = ro02 * ((*gpu_def).beta2) * S2;
        F1S = (-1) * ro01 * (1 + ((*gpu_def).beta1) * (P1 - P_atm));
        F2S = ro02 * (1 + ((*gpu_def).beta2) * (P1 + P_k - P_atm + (S2 * PkS)));

        det = F1P * F2S - F1S * F2P;

        DevArraysPtr.P1[i+j*localNx+k*localNx*((*gpu_def).Ny)] = P1 - (1 / det) * (F2S * F1 - F1S * F2);
        DevArraysPtr.S2[i+j*localNx+k*localNx*((*gpu_def).Ny)] = S2 - (1 / det) * (F1P * F2 - F2P * F1);
    }
}

// Расчет давления воды P1 и насыщенности NAPL S2 во всех точках сетки
void P1_S2_calculation(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr, consts def, int localNx, int rank, int size, int blocksX, int blocksY, int blocksZ)
{
	Newton_method_kernel<<<dim3(blocksX,blocksY*blocksZ), dim3(BlockNX,BlockNY,BlockNZ)>>>(DevArraysPtr,localNx); 
	checkErrors("assign P1 and S2");
}

// Граничные условия на S2
__global__ void S2_boundary_kernel(ptr_Arrays DevArraysPtr, int localNx, int rank, int size) 
{
	int blockIdxz=blockIdx.y / BlockNY;
	int blockIdxy=blockIdx.y % BlockNY;
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	int k=threadIdx.z+blockIdxz*blockDim.z;
	int j=threadIdx.y+blockIdxy*blockDim.y;

	if ((i<localNx) && (j<((*gpu_def).Ny)) && (k<((*gpu_def).Nz)) && (device_is_active_point(i, localNx, rank, size)==1))
	{
		if ((i == 0) && (((*gpu_def).NX)>2))
		{
		   DevArraysPtr.S2[i+j*localNx+k*localNx*((*gpu_def).Ny)] = DevArraysPtr.S2[i+1+j*localNx+k*localNx*((*gpu_def).Ny)];
		}

		if ((i == localNx - 1) && (((*gpu_def).NX)>2))
		{
			DevArraysPtr.S2[i+j*localNx+k*localNx*((*gpu_def).Ny)] = DevArraysPtr.S2[i-1+j*localNx+k*localNx*((*gpu_def).Ny)];
		}

		if ((j == ((*gpu_def).Ny) - 1) && (((*gpu_def).Ny)>2))
		{
			DevArraysPtr.S2[i+j*localNx+k*localNx*((*gpu_def).Ny)] = DevArraysPtr.S2[i+(j-1)*localNx+k*localNx*((*gpu_def).Ny)];
		}

		if ((j==0) && (((*gpu_def).Ny)>2))
		{
			int I=device_i_to_I(i,rank,size, gpu_def[0]);
			if ((I>=((*gpu_def).NX)/2-((*gpu_def).source)) && (I<=((*gpu_def).NX)/2+((*gpu_def).source)) && (k>=((*gpu_def).Nz)/2-((*gpu_def).source)) && (k<=((*gpu_def).Nz)/2+((*gpu_def).source)))
				DevArraysPtr.S2[i+j*localNx+k*localNx*((*gpu_def).Ny)] = S2_gr;
			else
				DevArraysPtr.S2[i+j*localNx+k*localNx*((*gpu_def).Ny)] = 0;
		}

		if ((k == 0) && (((*gpu_def).Nz)>2))
		{
			DevArraysPtr.S2[i+j*localNx+k*localNx*((*gpu_def).Ny)] = DevArraysPtr.S2[i+j*localNx+(k+1)*localNx*((*gpu_def).Ny)];
		}

		if ((k == ((*gpu_def).Nz) - 1) && (((*gpu_def).Nz)>2))
		{
			DevArraysPtr.S2[i+j*localNx+k*localNx*((*gpu_def).Ny)] = DevArraysPtr.S2[i+j*localNx+(k-1)*localNx*((*gpu_def).Ny)];
		}
	}
}

// Граничные условия на P1
__global__ void P1_boundary_kernel(ptr_Arrays DevArraysPtr, int localNx, int rank, int size) 
{
	int blockIdxz=blockIdx.y / BlockNY;
	int blockIdxy=blockIdx.y % BlockNY;
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	int k=threadIdx.z+blockIdxz*blockDim.z;
	int j=threadIdx.y+blockIdxy*blockDim.y;

	if ((i<localNx) && (j<((*gpu_def).Ny)) && (k<((*gpu_def).Nz)) && (device_is_active_point(i, localNx, rank, size)==1))
	{
		if ((i == 0) && (((*gpu_def).NX)>2))
		{
			DevArraysPtr.P1[i+j*localNx+k*localNx*((*gpu_def).Ny)] = DevArraysPtr.P1[i+1+j*localNx+k*localNx*((*gpu_def).Ny)]; 
		}

		if ((i == localNx - 1) && (((*gpu_def).NX)>2))
		{
			DevArraysPtr.P1[i+j*localNx+k*localNx*((*gpu_def).Ny)] = DevArraysPtr.P1[i-1+j*localNx+k*localNx*((*gpu_def).Ny)];
		}

		if ((j == ((*gpu_def).Ny) - 1) && (((*gpu_def).Ny)>2))
		{
			DevArraysPtr.P1[i+j*localNx+k*localNx*((*gpu_def).Ny)] = DevArraysPtr.P1[i+(j-1)*localNx+k*localNx*((*gpu_def).Ny)] + DevArraysPtr.ro1[i+localNx*1] * g_const * ((*gpu_def).h2);
		}

		if ((j==0) && (((*gpu_def).Ny)>2))
		{
			DevArraysPtr.P1[i+j*localNx+k*localNx*((*gpu_def).Ny)] = P_atm;
		}

		if ((k == 0) && (((*gpu_def).Nz)>2))
		{
			DevArraysPtr.P1[i+j*localNx+k*localNx*((*gpu_def).Ny)] = DevArraysPtr.P1[i+j*localNx+(k+1)*localNx*((*gpu_def).Ny)]; 
		}

		if ((k == ((*gpu_def).Nz) - 1) && (((*gpu_def).Nz)>2))
		{
			DevArraysPtr.P1[i+j*localNx+k*localNx*((*gpu_def).Ny)] = DevArraysPtr.P1[i+j*localNx+(k-1)*localNx*((*gpu_def).Ny)];
		}
	}
}

// Применение граничных условий
void boundary_conditions(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr, int localNx, int rank, int size, int blocksX, int blocksY, int blocksZ, consts def)
{
	S2_boundary_kernel<<<dim3(blocksX,blocksY*blocksZ), dim3(BlockNX,BlockNY,BlockNZ)>>>(DevArraysPtr,localNx,rank,size); 
	checkErrors("assign S2");
	P1_boundary_kernel<<<dim3(blocksX,blocksY*blocksZ), dim3(BlockNX,BlockNY,BlockNZ)>>>(DevArraysPtr,localNx,rank,size); 
	checkErrors("assign P1");
}	

// Функция загрузки данных в память хоста
void load_data_to_host(double* HostArrayPtr, double* DevArrayPtr, int localNx, consts def)
{
	cudaMemcpy( HostArrayPtr, DevArrayPtr, localNx*(def.Ny)*(def.Nz)*sizeof(double), cudaMemcpyDeviceToHost );
	checkErrors("copy data to host");
}

// Функция загрузки данных типа double в память ускорителя
void load_data_to_device(double* HostArrayPtr, double* DevArrayPtr, int localNx, consts def)
{
	cudaMemcpy( DevArrayPtr, HostArrayPtr, localNx*(def.Ny)*(def.Nz)*sizeof(double), cudaMemcpyHostToDevice );
	checkErrors("copy double data to device");
}

// Функция загрузки данных типа int в память ускорителя
void load_data_to_device_int(int* HostArrayPtr, int* DevArrayPtr, int localNx, consts def)
{
	cudaMemcpy( DevArrayPtr, HostArrayPtr, localNx*(def.Ny)*(def.Nz)*sizeof(int), cudaMemcpyHostToDevice );
	checkErrors("copy int data to device");
}

// Выделение памяти ускорителя под массив точек расчетной области
void device_memory_alloc(ptr_Arrays* ArraysPtr, double** DevBuffer, int localNx, consts def)
{
	cudaMalloc((void**) DevBuffer,  2 * (def.Ny) * (def.Nz) * sizeof(double));

	cudaMalloc((void**) &((*ArraysPtr).x),  localNx * (def.Ny) * (def.Nz) * sizeof(double));
	cudaMalloc((void**) &((*ArraysPtr).y),  localNx * (def.Ny) * (def.Nz) * sizeof(double));
	cudaMalloc((void**) &((*ArraysPtr).z),  localNx * (def.Ny) * (def.Nz) * sizeof(double));
	cudaMalloc((void**) &((*ArraysPtr).P1),  localNx * (def.Ny) * (def.Nz) * sizeof(double));
	cudaMalloc((void**) &((*ArraysPtr).P2),  localNx * (def.Ny) * (def.Nz) * sizeof(double));
	cudaMalloc((void**) &((*ArraysPtr).S2),  localNx * (def.Ny) * (def.Nz) * sizeof(double));
	cudaMalloc((void**) &((*ArraysPtr).ro1),  localNx * (def.Ny) * (def.Nz) * sizeof(double));
	cudaMalloc((void**) &((*ArraysPtr).ro2),  localNx * (def.Ny) * (def.Nz) * sizeof(double));
	cudaMalloc((void**) &((*ArraysPtr).u1x),  localNx * (def.Ny) * (def.Nz) * sizeof(double));
	cudaMalloc((void**) &((*ArraysPtr).u1y),  localNx * (def.Ny) * (def.Nz) * sizeof(double));
	cudaMalloc((void**) &((*ArraysPtr).u1z),  localNx * (def.Ny) * (def.Nz) * sizeof(double));
	cudaMalloc((void**) &((*ArraysPtr).u2x),  localNx * (def.Ny) * (def.Nz) * sizeof(double));
	cudaMalloc((void**) &((*ArraysPtr).u2y),  localNx * (def.Ny) * (def.Nz) * sizeof(double));
	cudaMalloc((void**) &((*ArraysPtr).u2z),  localNx * (def.Ny) * (def.Nz) * sizeof(double));
	cudaMalloc((void**) &((*ArraysPtr).Xi1),  localNx * (def.Ny) * (def.Nz) * sizeof(double));
	cudaMalloc((void**) &((*ArraysPtr).Xi2),  localNx * (def.Ny) * (def.Nz) * sizeof(double));
	cudaMalloc((void**) &((*ArraysPtr).roS1),  localNx * (def.Ny) * (def.Nz) * sizeof(double));
	cudaMalloc((void**) &((*ArraysPtr).roS2),  localNx * (def.Ny) * (def.Nz) * sizeof(double));
	cudaMalloc((void**) &((*ArraysPtr).roS1_old),  localNx * (def.Ny) * (def.Nz) * sizeof(double));
	cudaMalloc((void**) &((*ArraysPtr).roS2_old),  localNx * (def.Ny) * (def.Nz) * sizeof(double));
	cudaMalloc((void**) &((*ArraysPtr).media),  localNx * (def.Ny) * (def.Nz) * sizeof(int));

	checkErrors("memory allocation");
}

// Освобожение памяти ускорителя из под массива точек расчетной области
void device_memory_free(ptr_Arrays DevArraysPtr, double* DevBuffer)
{
	cudaFree (DevBuffer);
	cudaFree (DevArraysPtr.x);
	cudaFree (DevArraysPtr.y);
	cudaFree (DevArraysPtr.z);
	cudaFree (DevArraysPtr.P1);
	cudaFree (DevArraysPtr.P2);
	cudaFree (DevArraysPtr.S2);
	cudaFree (DevArraysPtr.ro1);
	cudaFree (DevArraysPtr.ro2);
	cudaFree (DevArraysPtr.u1x);
	cudaFree (DevArraysPtr.u1y);
	cudaFree (DevArraysPtr.u1z);
	cudaFree (DevArraysPtr.u2x);
	cudaFree (DevArraysPtr.u2y);
	cudaFree (DevArraysPtr.u2z);
	cudaFree (DevArraysPtr.Xi1);
	cudaFree (DevArraysPtr.Xi2);
	cudaFree (DevArraysPtr.roS1);
	cudaFree (DevArraysPtr.roS2);
	cudaFree (DevArraysPtr.roS1_old);
	cudaFree (DevArraysPtr.roS2_old);
	cudaFree (DevArraysPtr.media);

	checkErrors("memory reliase");
}

// Инициализация ускорителя
// Расчет происходит на ускорителе, номер которого равен
// номеру запускающего процессора
void Device_Initialize(int rank, int* blocksX, int* blocksY, int* blocksZ, int localNx, consts def)
{
	// Если 3 ускорителя на одном узле с большим количеством ядер
	int device=rank%3;
	cudaSetDevice(device);
	printf("CUDA initialized.\n");

	// Количество запускаемых блоков
	// Если число точек сетки не кратно размеру блока,
	// то количество блоков будет на 1 больше.
	*blocksX=localNx/BlockNX;
	if (localNx%BlockNX!=0)
		(*blocksX)++;
	*blocksY=(def.Ny)/BlockNY;
	if ((def.Ny)%BlockNY!=0)
		(*blocksY)++;
	*blocksZ=(def.Nz)/BlockNZ;
	if ((def.Nz)%BlockNZ!=0)
		(*blocksZ)++;

	cudaMemcpyToSymbol ( gpu_def, &def, sizeof ( consts ), 0, cudaMemcpyHostToDevice );
	checkErrors("constant memory copy");

	int deviceCount;
    cudaDeviceProp devProp;
    cudaGetDeviceCount ( &deviceCount );

        cudaGetDeviceProperties ( &devProp, device );
        printf ( "Device %d\n", device );
        printf ( "Compute capability : %d.%d\n", devProp.major, devProp.minor );
        printf ( "Name : %s\n", devProp.name );
        printf ( "Total Global Memory : %ld\n", devProp.totalGlobalMem );
        printf ( "Shared memory per block: %d\n", devProp.sharedMemPerBlock );
        printf ( "Registers per block : %d\n", devProp.regsPerBlock );
        printf ( "Warp size : %d\n", devProp.warpSize );
        printf ( "Max threads per block : %d\n", devProp.maxThreadsPerBlock );
        printf ( "Total constant memory : %d\n\n", devProp.totalConstMem );

		// Максимальный размер расчетной сетки для ускорителя
		// 21 - количество параметров в точке
		printf ( "Total NAPL_Filtration grid size : %d\n\n", devProp.totalGlobalMem/(21*sizeof(double)) );

		// localNX+2 потому что 2NyNz на буфер обмена выделяется
		if ((localNx+2)*(def.Ny)*(def.Nz) > (devProp.totalGlobalMem/(21*sizeof(double))))
			printf ("\nError! Not enough memory at GPU, rank=%d\n",rank);
}


__global__ void load_exchange_data_kernel(double* DevArrayPtr, double* DevBuffer, int localNx)
{
	int j=threadIdx.x+blockIdx.x*blockDim.x;
	int k=threadIdx.y+blockIdx.y*blockDim.y;

	if (j<(*gpu_def).Ny && k<(*gpu_def).Nz)
	{
		DevBuffer[j+((*gpu_def).Ny)*k]=DevArrayPtr[1+localNx*j+localNx*((*gpu_def).Ny)*k];
		DevBuffer[j+((*gpu_def).Ny)*k+((*gpu_def).Ny)*((*gpu_def).Nz)]=DevArrayPtr[localNx-2+localNx*j+localNx*((*gpu_def).Ny)*k];
	}
}


void load_exchange_data(double* HostArrayPtr, double* DevArrayPtr, double* HostBuffer, double* DevBuffer, int localNx, int blocksY, int blocksZ, int rank, int size, consts def)
{
	load_exchange_data_kernel<<<dim3(blocksY,blocksZ), dim3(BlockNY,BlockNZ)>>>(DevArrayPtr, DevBuffer, localNx); 
	checkErrors("load_exchange_data");
	cudaMemcpy( HostBuffer, DevBuffer, 2*(def.Ny)*(def.Nz)*sizeof(double), cudaMemcpyDeviceToHost );
	checkErrors("copy data to host");
}

__global__ void save_exchange_data_kernel(double* DevArrayPtr, double* DevBuffer, int localNx, int rank, int size)
{
	int j=threadIdx.x+blockIdx.x*blockDim.x;
	int k=threadIdx.y+blockIdx.y*blockDim.y;

	if (j<(*gpu_def).Ny && k<(*gpu_def).Nz)
	{
		if (rank!=0)
			DevArrayPtr[localNx*j+localNx*((*gpu_def).Ny)*k]=DevBuffer[j+((*gpu_def).Ny)*k];
		if (rank!=size-1)
			DevArrayPtr[localNx-1+localNx*j+localNx*((*gpu_def).Ny)*k]=DevBuffer[j+((*gpu_def).Ny)*k+((*gpu_def).Ny)*((*gpu_def).Nz)];
	}
}

void save_exchange_data(double* HostArrayPtr, double* DevArrayPtr, double* HostBuffer, double* DevBuffer, int localNx, int blocksY, int blocksZ, int rank, int size, consts def)
{
	cudaMemcpy( DevBuffer, HostBuffer, 2*(def.Ny)*(def.Nz)*sizeof(double), cudaMemcpyHostToDevice );
	checkErrors("copy data to device");
	save_exchange_data_kernel<<<dim3(blocksY,blocksZ), dim3(BlockNY,BlockNZ)>>>(DevArrayPtr, DevBuffer, localNx, rank, size); 
	checkErrors("save_exchange_data");
}