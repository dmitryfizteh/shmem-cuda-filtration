#include "defines.h"

void ro_P2_Xi_calculation(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr, consts def, int localNx, int rank, int size, int blocksX, int blocksY, int blocksZ)
{
	for(int i=0;i<localNx;i++)
		for(int j=0;j<(def.Ny);j++)
			for(int k=0;k<(def.Nz);k++)
			{
				if(is_active_point(i, localNx, rank, size))
				{
					assign_P2_Xi1_Xi2(HostArraysPtr,i,j,k,localNx,def);
					assign_ro1_ro2(HostArraysPtr,i,j,k,localNx,def);
				}
			}
}

void u_calculation(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr, int localNx, int rank, int size, int blocksX, int blocksY, int blocksZ, consts def)
{
	for(int i=0;i<localNx;i++)
		for(int j=0;j<(def.Ny);j++)
			for(int k=0;k<(def.Nz);k++)
				if(is_active_point(i, localNx, rank, size))
					assign_u(HostArraysPtr,i,j,k,localNx,def);
}

void roS_calculation(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr, consts def, double t, int localNx, int rank, int size, int blocksX, int blocksY, int blocksZ)
{
	for(int i=0;i<localNx;i++)
		for(int j=0;j<(def.Ny);j++)
			for(int k=0;k<(def.Nz);k++)
				if(is_active_point(i, localNx, rank, size))
					assign_rS_nr(HostArraysPtr,t,i,j,k,localNx,def);
}

void P1_S2_calculation(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr, consts def, int localNx, int rank, int size, int blocksX, int blocksY, int blocksZ)
{
	for(int i=0;i<localNx;i++)
		for(int j=0;j<(def.Ny);j++)
			for(int k=0;k<(def.Nz);k++)
				if(is_active_point(i, localNx, rank, size))
					Newton(HostArraysPtr,i,j,k,localNx,def);
}

void boundary_conditions(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr, int localNx, int rank, int size, int blocksX, int blocksY, int blocksZ, consts def)
{
	for(int i=0;i<localNx;i++)
		for(int j=0;j<(def.Ny);j++)
			for(int k=0;k<(def.Nz);k++)
				if(is_active_point(i, localNx, rank, size))
				{
					Border_S2(HostArraysPtr,i,j,k,localNx,rank,size,def);
					Border_P1(HostArraysPtr,i,j,k,localNx,def);
				}
}	

void assign_P2_Xi1_Xi2(ptr_Arrays HostArraysPtr, int i, int j, int k, int localNx, consts def)
{
	int media = HostArraysPtr.media[i+j*localNx+k*localNx*(def.Ny)];
	double S_e = (1. - HostArraysPtr.S2[i+j*localNx+k*localNx*(def.Ny)] - def.S_wr[media]) / (1. - def.S_wr[media]);
	double k1 = pow(S_e, (2. + 3. * def.lambda[media]) / def.lambda[media]);
	double k2 = (1. - S_e) * (1. - S_e) * (1 - pow(S_e, (2. + def.lambda[media]) / def.lambda[media]));
	double P_k = def.P_d[media] * pow((1. - HostArraysPtr.S2[i+j*localNx+k*localNx*(def.Ny)] - def.S_wr[media]) / (1. - def.S_wr[media]), -1. / def.lambda[media]);

	HostArraysPtr.P2[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.P1[i+j*localNx+k*localNx*(def.Ny)] + P_k;
	HostArraysPtr.Xi1[i+j*localNx+k*localNx*(def.Ny)] = -1. * def.K[media] * k1 / mu1;
	HostArraysPtr.Xi2[i+j*localNx+k*localNx*(def.Ny)] = -1. * def.K[media] * k2 / mu2;
}

void assign_ro1_ro2(ptr_Arrays HostArraysPtr, int i, int j, int k, int localNx, consts def)
{
	HostArraysPtr.ro1[i+j*localNx+k*localNx*(def.Ny)] = ro01 * (1. + (def.beta1) * (HostArraysPtr.P1[i+j*localNx+k*localNx*(def.Ny)] - P_atm));
	HostArraysPtr.ro2[i+j*localNx+k*localNx*(def.Ny)] = ro02 * (1. + (def.beta2) * (HostArraysPtr.P2[i+j*localNx+k*localNx*(def.Ny)] - P_atm));
}

// (!3D)
void assign_u(ptr_Arrays HostArraysPtr, int i, int j, int k, int localNx, consts def)
{
	//std::cout << i << " " << j << " " << k << "\n";
	if ((def.NX)>2)
	{
		if (i == 0)
		{
			HostArraysPtr.u1x[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.Xi1[i+j*localNx+k*localNx*(def.Ny)] * (HostArraysPtr.P1[i+1+j*localNx+k*localNx*(def.Ny)] - HostArraysPtr.P1[i+j*localNx+k*localNx*(def.Ny)]) / (def.h1);
			HostArraysPtr.u2x[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.Xi2[i+j*localNx+k*localNx*(def.Ny)] * (HostArraysPtr.P2[i+1+j*localNx+k*localNx*(def.Ny)] - HostArraysPtr.P2[i+j*localNx+k*localNx*(def.Ny)]) / (def.h1);
		}
		else
		{
			if (i == localNx - 1)
			{
				HostArraysPtr.u1x[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.Xi1[i+j*localNx+k*localNx*(def.Ny)] * (HostArraysPtr.P1[i+j*localNx+k*localNx*(def.Ny)] - HostArraysPtr.P1[i-1+j*localNx+k*localNx*(def.Ny)]) / (def.h1);
				HostArraysPtr.u2x[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.Xi2[i+j*localNx+k*localNx*(def.Ny)] * (HostArraysPtr.P2[i+j*localNx+k*localNx*(def.Ny)] - HostArraysPtr.P2[i-1+j*localNx+k*localNx*(def.Ny)]) / (def.h1);
			}
			else
			{
				HostArraysPtr.u1x[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.Xi1[i+j*localNx+k*localNx*(def.Ny)] * ((HostArraysPtr.P1[i+1+j*localNx+k*localNx*(def.Ny)] - HostArraysPtr.P1[i-1+j*localNx+k*localNx*(def.Ny)]) / (2. * (def.h1)));
				HostArraysPtr.u2x[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.Xi2[i+j*localNx+k*localNx*(def.Ny)] * ((HostArraysPtr.P2[i+1+j*localNx+k*localNx*(def.Ny)] - HostArraysPtr.P2[i-1+j*localNx+k*localNx*(def.Ny)]) / (2. * (def.h1)));
			}
		}
	}
	else
	{
		HostArraysPtr.u1x[i+j*localNx+k*localNx*(def.Ny)] = 0;
		HostArraysPtr.u2x[i+j*localNx+k*localNx*(def.Ny)] = 0;
	}

	if ((def.Ny)>2)
	{
		if (j == 0)
		{
			HostArraysPtr.u1y[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.Xi1[i+j*localNx+k*localNx*(def.Ny)] * ((HostArraysPtr.P1[i+(j+1)*localNx+k*localNx*(def.Ny)] - HostArraysPtr.P1[i+j*localNx+k*localNx*(def.Ny)]) / (def.h2) - HostArraysPtr.ro1[i+j*localNx+k*localNx*(def.Ny)] * g_const);
			HostArraysPtr.u2y[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.Xi2[i+j*localNx+k*localNx*(def.Ny)] * ((HostArraysPtr.P2[i+(j+1)*localNx+k*localNx*(def.Ny)] - HostArraysPtr.P2[i+j*localNx+k*localNx*(def.Ny)]) / (def.h2) - HostArraysPtr.ro2[i+j*localNx+k*localNx*(def.Ny)] * g_const);
		}
		else
		{
			if (j == (def.Ny) - 1)
			{
				HostArraysPtr.u1y[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.Xi1[i+j*localNx+k*localNx*(def.Ny)] * ((HostArraysPtr.P1[i+j*localNx+k*localNx*(def.Ny)] - HostArraysPtr.P1[i+(j-1)*localNx+k*localNx*(def.Ny)]) / (def.h2) - HostArraysPtr.ro1[i+j*localNx+k*localNx*(def.Ny)] * g_const);
				HostArraysPtr.u2y[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.Xi2[i+j*localNx+k*localNx*(def.Ny)] * ((HostArraysPtr.P2[i+j*localNx+k*localNx*(def.Ny)] - HostArraysPtr.P2[i+(j-1)*localNx+k*localNx*(def.Ny)]) / (def.h2) - HostArraysPtr.ro2[i+j*localNx+k*localNx*(def.Ny)] * g_const);
			}
			else
			{
				HostArraysPtr.u1y[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.Xi1[i+j*localNx+k*localNx*(def.Ny)] * ((HostArraysPtr.P1[i+(j+1)*localNx+k*localNx*(def.Ny)] - HostArraysPtr.P1[i+(j-1)*localNx+k*localNx*(def.Ny)]) / (2. * (def.h2)) - HostArraysPtr.ro1[i+j*localNx+k*localNx*(def.Ny)] * g_const);
				HostArraysPtr.u2y[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.Xi2[i+j*localNx+k*localNx*(def.Ny)] * ((HostArraysPtr.P2[i+(j+1)*localNx+k*localNx*(def.Ny)] - HostArraysPtr.P2[i+(j-1)*localNx+k*localNx*(def.Ny)]) / (2. * (def.h2)) - HostArraysPtr.ro2[i+j*localNx+k*localNx*(def.Ny)] * g_const);
			}
		}
	}
	else
	{
		HostArraysPtr.u1y[i+j*localNx+k*localNx*(def.Ny)] = 0;
		HostArraysPtr.u2y[i+j*localNx+k*localNx*(def.Ny)] = 0;
	}

	if ((def.Nz)>2)
	{
		if (k == 0)
		{
			HostArraysPtr.u1z[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.Xi1[i+j*localNx+k*localNx*(def.Ny)] * (HostArraysPtr.P1[i+localNx*j+(k+1)*localNx*(def.Ny)] - HostArraysPtr.P1[i+j*localNx+k*localNx*(def.Ny)]) / (def.h3);
			HostArraysPtr.u2z[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.Xi2[i+j*localNx+k*localNx*(def.Ny)] * (HostArraysPtr.P2[i+localNx*j+(k+1)*localNx*(def.Ny)] - HostArraysPtr.P2[i+j*localNx+k*localNx*(def.Ny)]) / (def.h3);
		}
		else
		{
			if (k == (def.Nz) - 1)
			{
				HostArraysPtr.u1z[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.Xi1[i+j*localNx+k*localNx*(def.Ny)] * (HostArraysPtr.P1[i+j*localNx+k*localNx*(def.Ny)] - HostArraysPtr.P1[i+localNx*j+(k-1)*localNx*(def.Ny)]) / (def.h3);
				HostArraysPtr.u2z[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.Xi2[i+j*localNx+k*localNx*(def.Ny)] * (HostArraysPtr.P2[i+j*localNx+k*localNx*(def.Ny)] - HostArraysPtr.P2[i+localNx*j+(k-1)*localNx*(def.Ny)]) / (def.h3);
			}
			else
			{
				HostArraysPtr.u1z[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.Xi1[i+j*localNx+k*localNx*(def.Ny)] * ((HostArraysPtr.P1[i+localNx*j+(k+1)*localNx*(def.Ny)] - HostArraysPtr.P1[i+localNx*j+(k-1)*localNx*(def.Ny)]) / (2. * (def.h3)));
				HostArraysPtr.u2z[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.Xi2[i+j*localNx+k*localNx*(def.Ny)] * ((HostArraysPtr.P2[i+localNx*j+(k+1)*localNx*(def.Ny)] - HostArraysPtr.P2[i+localNx*j+(k-1)*localNx*(def.Ny)]) / (2. * (def.h3)));
			}
		}
	}
	else
	{
		HostArraysPtr.u1z[i+j*localNx+k*localNx*(def.Ny)] = 0;
		HostArraysPtr.u2z[i+j*localNx+k*localNx*(def.Ny)] = 0;
	}
}

void assign_rS(ptr_Arrays HostArraysPtr, double t, int i, int j, int k, int localNx, consts def)
{
	if ((i!=0) && (i!=localNx-1) && (j!=0) && (j!=(def.Ny)-1) && (((k!=0) && (k!=(def.Nz)-1)) || ((def.Nz)<2)))
	{
		int media = HostArraysPtr.media[i+j*localNx+k*localNx*(def.Ny)];

		HostArraysPtr.roS1[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.ro1[i+j*localNx+k*localNx*(def.Ny)] * (1 - HostArraysPtr.S2[i+j*localNx+k*localNx*(def.Ny)]);
		HostArraysPtr.roS2[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.ro2[i+j*localNx+k*localNx*(def.Ny)] * HostArraysPtr.S2[i+j*localNx+k*localNx*(def.Ny)];

		double divgrad1, divgrad2, Tx1, Ty1, Tz1, Tx2, Ty2, Tz2, A1=0, A2=0;

		if ((def.Nz)<2)
		{
			divgrad1=0;
			divgrad2=0;
			Tz1=0;
			Tz2=0;
		}
		else
		{
			divgrad1 = (def.m[media] * (def.l_w) * (def.c) / 2.) * (HostArraysPtr.ro1[i+j*localNx+(k+1)*localNx*(def.Ny)] * (1. - HostArraysPtr.S2[i+j*localNx+(k+1)*localNx*(def.Ny)]) - 2 * HostArraysPtr.ro1[i+j*localNx+k*localNx*(def.Ny)] * (1. - HostArraysPtr.S2[i+j*localNx+k*localNx*(def.Ny)]) + HostArraysPtr.ro1[i+j*localNx+(k-1)*localNx*(def.Ny)] * (1. - HostArraysPtr.S2[i+j*localNx+(k-1)*localNx*(def.Ny)])) / ((def.h3) * (def.h3));
			divgrad2 = (def.m[media] * (def.l_n) * (def.c) / 2.) * (HostArraysPtr.ro2[i+j*localNx+(k+1)*localNx*(def.Ny)] * HostArraysPtr.S2[i+j*localNx+(k+1)*localNx*(def.Ny)] - 2 * HostArraysPtr.ro2[i+j*localNx+k*localNx*(def.Ny)] * HostArraysPtr.S2[i+j*localNx+k*localNx*(def.Ny)] + HostArraysPtr.ro2[i+j*localNx+(k-1)*localNx*(def.Ny)] * (HostArraysPtr.S2[i+j*localNx+(k-1)*localNx*(def.Ny)])) / ((def.h3) * (def.h3));
			Tz1 = (HostArraysPtr.ro1[i+1+j*localNx+(k+1)*localNx*(def.Ny)] * HostArraysPtr.u1x[i+1+j*localNx+(k+1)*localNx*(def.Ny)] - HostArraysPtr.ro1[i+j*localNx+(k-1)*localNx*(def.Ny)] * HostArraysPtr.u1x[i+j*localNx+(k-1)*localNx*(def.Ny)]) / (2. * (def.h3));
			Tz2 = (HostArraysPtr.ro2[i+j*localNx+(k+1)*localNx*(def.Ny)] * HostArraysPtr.u2y[i+j*localNx+(k+1)*localNx*(def.Ny)] - HostArraysPtr.ro2[i+j*localNx+(k-1)*localNx*(def.Ny)] * HostArraysPtr.u2y[i+j*localNx+(k-1)*localNx*(def.Ny)]) / (2. * (def.h3));
		}

		divgrad1 += (def.m[media] * (def.l_w) * (def.c) / 2.) *
			((HostArraysPtr.ro1[i+1+j*localNx+k*localNx*(def.Ny)] * (1. - HostArraysPtr.S2[i+1+j*localNx+k*localNx*(def.Ny)]) - 2 * HostArraysPtr.ro1[i+j*localNx+k*localNx*(def.Ny)] * (1. - HostArraysPtr.S2[i+j*localNx+k*localNx*(def.Ny)]) + HostArraysPtr.ro1[i-1+j*localNx+k*localNx*(def.Ny)] * (1. - HostArraysPtr.S2[i-1+j*localNx+k*localNx*(def.Ny)])) / ((def.h1) * (def.h1)) +
			(HostArraysPtr.ro1[i+(j+1)*localNx+k*localNx*(def.Ny)] * (1. - HostArraysPtr.S2[i+(j+1)*localNx+k*localNx*(def.Ny)]) - 2 * HostArraysPtr.ro1[i+j*localNx+k*localNx*(def.Ny)] * (1. - HostArraysPtr.S2[i+j*localNx+k*localNx*(def.Ny)]) + HostArraysPtr.ro1[i+(j-1)*localNx+k*localNx*(def.Ny)] * (1. - HostArraysPtr.S2[i+(j-1)*localNx+k*localNx*(def.Ny)])) / ((def.h2) * (def.h2)));

		divgrad2 += (def.m[media] * (def.l_n) * (def.c) / 2.) *
			((HostArraysPtr.ro2[i+1+j*localNx+k*localNx*(def.Ny)] * HostArraysPtr.S2[i+1+j*localNx+k*localNx*(def.Ny)] - 2 * HostArraysPtr.ro2[i+j*localNx+k*localNx*(def.Ny)] * HostArraysPtr.S2[i+j*localNx+k*localNx*(def.Ny)] + HostArraysPtr.ro2[i-1+j*localNx+k*localNx*(def.Ny)] * (HostArraysPtr.S2[i-1+j*localNx+k*localNx*(def.Ny)])) / ((def.h1) * (def.h1)) +
			(HostArraysPtr.ro2[i+(j+1)*localNx+k*localNx*(def.Ny)] * HostArraysPtr.S2[i+(j+1)*localNx+k*localNx*(def.Ny)] - 2 * HostArraysPtr.ro2[i+j*localNx+k*localNx*(def.Ny)] * HostArraysPtr.S2[i+j*localNx+k*localNx*(def.Ny)] + HostArraysPtr.ro2[i+(j-1)*localNx+k*localNx*(def.Ny)] * (HostArraysPtr.S2[i+(j-1)*localNx+k*localNx*(def.Ny)])) / ((def.h2) * (def.h2)));

		Tx1 = (HostArraysPtr.ro1[i+1+j*localNx+k*localNx*(def.Ny)] * HostArraysPtr.u1x[i+1+j*localNx+k*localNx*(def.Ny)] - HostArraysPtr.ro1[i-1+j*localNx+k*localNx*(def.Ny)] * HostArraysPtr.u1x[i-1+j*localNx+k*localNx*(def.Ny)]) / (2. * (def.h1));
		Ty1 = (HostArraysPtr.ro1[i+(j+1)*localNx+k*localNx*(def.Ny)] * HostArraysPtr.u1y[i+(j+1)*localNx+k*localNx*(def.Ny)] - HostArraysPtr.ro1[i+(j-1)*localNx+k*localNx*(def.Ny)] * HostArraysPtr.u1y[i+(j-1)*localNx+k*localNx*(def.Ny)]) / (2. * (def.h2));
		Tx2 = (HostArraysPtr.ro2[i+1+j*localNx+k*localNx*(def.Ny)] * HostArraysPtr.u2x[i+1+j*localNx+k*localNx*(def.Ny)] - HostArraysPtr.ro2[i-1+j*localNx+k*localNx*(def.Ny)] * HostArraysPtr.u2x[i-1+j*localNx+k*localNx*(def.Ny)]) / (2. * (def.h1));
		Ty2 = (HostArraysPtr.ro2[i+(j+1)*localNx+k*localNx*(def.Ny)] * HostArraysPtr.u2y[i+(j+1)*localNx+k*localNx*(def.Ny)] - HostArraysPtr.ro2[i+(j-1)*localNx+k*localNx*(def.Ny)] * HostArraysPtr.u2y[i+(j-1)*localNx+k*localNx*(def.Ny)]) / (2. * (def.h2));

		if (t < 2 * (def.dt))
		{
			A1 = HostArraysPtr.roS1[i+j*localNx+k*localNx*(def.Ny)] + ((def.dt) / def.m[media]) * (divgrad1 - Tx1 - Ty1 - Tz1);
			A2 = HostArraysPtr.roS2[i+j*localNx+k*localNx*(def.Ny)] + ((def.dt) / def.m[media]) * (divgrad2 - Tx2 - Ty2 - Tz2);
		}
		else
		{
			A1 = (2. * (def.dt) * (def.dt)) / (def.m[media] * ((def.dt) + 2. * (def.tau))) * (divgrad1 - Tx1 - Ty1 - Tz1 + (2. * HostArraysPtr.roS1[i+j*localNx+k*localNx*(def.Ny)] * def.m[media] * (def.tau)) / ((def.dt) * (def.dt)) + HostArraysPtr.roS1_old[i+j*localNx+k*localNx*(def.Ny)] * def.m[media] * ((def.dt) - 2. * (def.tau)) / (2. * (def.dt) * (def.dt)));
			A2 = (2. * (def.dt) * (def.dt)) / (def.m[media] * ((def.dt) + 2. * (def.tau))) * (divgrad2 - Tx2 - Ty2 - Tz2 + (2. * HostArraysPtr.roS2[i+j*localNx+k*localNx*(def.Ny)] * def.m[media] * (def.tau)) / ((def.dt) * (def.dt)) + HostArraysPtr.roS2_old[i+j*localNx+k*localNx*(def.Ny)] * def.m[media] * ((def.dt) - 2. * (def.tau)) / (2. * (def.dt) * (def.dt)));
		}
		HostArraysPtr.roS1_old[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.roS1[i+j*localNx+k*localNx*(def.Ny)];
		HostArraysPtr.roS2_old[i+j*localNx+k*localNx*(def.Ny)]= HostArraysPtr.roS2[i+j*localNx+k*localNx*(def.Ny)];
		HostArraysPtr.roS1[i+j*localNx+k*localNx*(def.Ny)] = A1;
		HostArraysPtr.roS2[i+j*localNx+k*localNx*(def.Ny)] = A2;
	}
}

void assign_rS_nr(ptr_Arrays HostArraysPtr, double t, int i, int j, int k, int localNx, consts def)
{
	if ((i!=0) && (i!=localNx-1) && (j!=0) && (j!=(def.Ny)-1) && (((k!=0) && (k!=(def.Nz)-1)) || ((def.Nz)<2)))
	{
		int media = HostArraysPtr.media[i+j*localNx+k*localNx*(def.Ny)];

		HostArraysPtr.roS1[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.ro1[i+j*localNx+k*localNx*(def.Ny)] * (1 - HostArraysPtr.S2[i+j*localNx+k*localNx*(def.Ny)]);
		HostArraysPtr.roS2[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.ro2[i+j*localNx+k*localNx*(def.Ny)] * HostArraysPtr.S2[i+j*localNx+k*localNx*(def.Ny)];

		double P1 = HostArraysPtr.P1[i+j*localNx+k*localNx*(def.Ny)];
		double P2 = HostArraysPtr.P2[i+j*localNx+k*localNx*(def.Ny)];

		double x1, x2, y1, y2, z1, z2, f1, f2, f3, g1, g2, g3, A1=0, A2=0;

		if ((def.Nz)<2)
		{
			f3=0;
			g3=0;
		}
		else
		{
			z2 = -(HostArraysPtr.P1[i+j*localNx+(k+1)*localNx*(def.Ny)] - P1)/def.h3;
			z1 = -(P1 - HostArraysPtr.P1[i+j*localNx+(k-1)*localNx*(def.Ny)])/def.h3;

			f3 = (((z2 + abs(z2))/2.0 - (z1 - abs(z1))/2.0)*(-1) * HostArraysPtr.Xi1[i+j*localNx+k*localNx*(def.Ny)] * HostArraysPtr.ro1[i+j*localNx+k*localNx*(def.Ny)] -
                      (z1 + abs(z1))/2.0*(-1)* HostArraysPtr.Xi1[i+j*localNx+(k-1)*localNx*(def.Ny)] * HostArraysPtr.ro1[i+j*localNx+(k-1)*localNx*(def.Ny)] +
                      (z2 - abs(z2))/2.0*(-1)* HostArraysPtr.Xi1[i+j*localNx+(k+1)*localNx*(def.Ny)] * HostArraysPtr.ro1[i+j*localNx+(k+1)*localNx*(def.Ny)])/def.h3;

			z2 = -(HostArraysPtr.P2[i+j*localNx+(k+1)*localNx*(def.Ny)] - P2)/def.h3;
			z1 = -(P2 - HostArraysPtr.P2[i+j*localNx+(k-1)*localNx*(def.Ny)])/def.h3;

			g3 = (((z2 + abs(z2))/2.0 - (z1 - abs(z1))/2.0)*(-1) * HostArraysPtr.Xi2[i+j*localNx+k*localNx*(def.Ny)] * HostArraysPtr.ro2[i+j*localNx+k*localNx*(def.Ny)] -
                      (z1 + abs(z1))/2.0*(-1)* HostArraysPtr.Xi2[i+j*localNx+(k-1)*localNx*(def.Ny)] * HostArraysPtr.ro2[i+j*localNx+(k-1)*localNx*(def.Ny)] +
                      (z2 - abs(z2))/2.0*(-1)* HostArraysPtr.Xi2[i+j*localNx+(k+1)*localNx*(def.Ny)] * HostArraysPtr.ro2[i+j*localNx+(k+1)*localNx*(def.Ny)])/def.h3;
		}

		x2 = -(HostArraysPtr.P1[i+1+j*localNx+k*localNx*(def.Ny)] - P1)/def.h1;
        x1 = -(P1 - HostArraysPtr.P1[i-1+j*localNx+k*localNx*(def.Ny)])/def.h1;

        y2 = -(HostArraysPtr.P1[i+(j+1)*localNx+k*localNx*(def.Ny)] - P1)/def.h2 + HostArraysPtr.ro1[i+j*localNx+k*localNx*(def.Ny)] * g_const;
        y1 = -(P1 - HostArraysPtr.P1[i+(j-1)*localNx+k*localNx*(def.Ny)])/def.h2 + HostArraysPtr.ro1[i+j*localNx+k*localNx*(def.Ny)] * g_const;

        f1 = (((x2 + abs(x2))/2.0 - (x1 - abs(x1))/2.0)*(-1) * HostArraysPtr.Xi1[i+j*localNx+k*localNx*(def.Ny)] * HostArraysPtr.ro1[i+j*localNx+k*localNx*(def.Ny)] -
                (x1 + abs(x1))/2.0*(-1)* HostArraysPtr.Xi1[i-1+j*localNx+k*localNx*(def.Ny)] * HostArraysPtr.ro1[i-1+j*localNx+k*localNx*(def.Ny)] +
                (x2 - abs(x2))/2.0*(-1)* HostArraysPtr.Xi1[i+1+j*localNx+k*localNx*(def.Ny)] * HostArraysPtr.ro1[i+1+j*localNx+k*localNx*(def.Ny)])/def.h1;

        f2 = (((y2 + abs(y2))/2.0 - (y1 - abs(y1))/2.0)*(-1)* HostArraysPtr.Xi1[i+j*localNx+k*localNx*(def.Ny)] * HostArraysPtr.ro1[i+j*localNx+k*localNx*(def.Ny)] -
                (y1 + abs(y1))/2.0*(-1)* HostArraysPtr.Xi1[i+(j-1)*localNx+k*localNx*(def.Ny)] * HostArraysPtr.ro1[i+(j-1)*localNx+k*localNx*(def.Ny)] +
                (y2 - abs(y2))/2.0*(-1)* HostArraysPtr.Xi1[i+(j+1)*localNx+k*localNx*(def.Ny)] * HostArraysPtr.ro1[i+(j+1)*localNx+k*localNx*(def.Ny)])/def.h2;


        x2 = -(HostArraysPtr.P2[i+1+j*localNx+k*localNx*(def.Ny)] - P2)/def.h1;
        x1 = -(P2 - HostArraysPtr.P2[i-1+j*localNx+k*localNx*(def.Ny)])/def.h1;

        y2 = -(HostArraysPtr.P2[i+(j+1)*localNx+k*localNx*(def.Ny)] - P2)/def.h2 + HostArraysPtr.ro2[i+j*localNx+k*localNx*(def.Ny)] * g_const;
        y1 = -(P2 - HostArraysPtr.P2[i+(j-1)*localNx+k*localNx*(def.Ny)])/def.h2 + HostArraysPtr.ro2[i+j*localNx+k*localNx*(def.Ny)] * g_const;

        g1 = (((x2 + abs(x2))/2.0 - (x1 - abs(x1))/2.0)*(-1) * HostArraysPtr.Xi2[i+j*localNx+k*localNx*(def.Ny)] * HostArraysPtr.ro2[i+j*localNx+k*localNx*(def.Ny)] -
                (x1 + abs(x1))/2.0*(-1)* HostArraysPtr.Xi2[i-1+j*localNx+k*localNx*(def.Ny)] * HostArraysPtr.ro2[i-1+j*localNx+k*localNx*(def.Ny)] +
                (x2 - abs(x2))/2.0*(-1)* HostArraysPtr.Xi2[i+1+j*localNx+k*localNx*(def.Ny)] * HostArraysPtr.ro2[i+1+j*localNx+k*localNx*(def.Ny)])/def.h1;

        g2 = (((y2 + abs(y2))/2.0 - (y1 - abs(y1))/2.0)*(-1)* HostArraysPtr.Xi2[i+j*localNx+k*localNx*(def.Ny)] * HostArraysPtr.ro2[i+j*localNx+k*localNx*(def.Ny)] -
                (y1 + abs(y1))/2.0*(-1)* HostArraysPtr.Xi2[i+(j-1)*localNx+k*localNx*(def.Ny)] * HostArraysPtr.ro2[i+(j-1)*localNx+k*localNx*(def.Ny)] +
                (y2 - abs(y2))/2.0*(-1)* HostArraysPtr.Xi2[i+(j+1)*localNx+k*localNx*(def.Ny)] * HostArraysPtr.ro2[i+(j+1)*localNx+k*localNx*(def.Ny)])/def.h2;

		A1 = HostArraysPtr.roS1[i+j*localNx+k*localNx*(def.Ny)] - (def.dt/def.m[media])*(f1 + f2 + f3);
        A2 = HostArraysPtr.roS2[i+j*localNx+k*localNx*(def.Ny)] - (def.dt/def.m[media])*(g1 + g2 + g3);

		HostArraysPtr.roS1_old[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.roS1[i+j*localNx+k*localNx*(def.Ny)];
		HostArraysPtr.roS2_old[i+j*localNx+k*localNx*(def.Ny)]= HostArraysPtr.roS2[i+j*localNx+k*localNx*(def.Ny)];
		HostArraysPtr.roS1[i+j*localNx+k*localNx*(def.Ny)] = A1;
		HostArraysPtr.roS2[i+j*localNx+k*localNx*(def.Ny)] = A2;
	}
}

void Newton(ptr_Arrays HostArraysPtr, int i, int j, int k, int localNx, consts def)
{
	if ((i!=0) && (i!=localNx-1) && (j!=0) && (j!=(def.Ny)-1) && (((k!=0) && (k!=(def.Nz)-1)) || ((def.Nz)<2)))
	{
		int media = HostArraysPtr.media[i+j*localNx+k*localNx*(def.Ny)];
		double S_e, P_k, AAA, F1, F2, PkS, F1P, F2P, F1S, F2S, det;

		for (int w=1;w<=NEWTON_ITERATIONS;w++)
		{
			S_e = (1 - HostArraysPtr.S2[i+j*localNx+k*localNx*(def.Ny)] - def.S_wr[media]) / (1 - def.S_wr[media]);
			P_k = def.P_d[media] * pow(S_e, (-1.) / def.lambda[media]);
			AAA = pow(S_e, (((-1.) / def.lambda[media]) - 1.));
			F1 = ro01 * (1. + (def.beta1) * (HostArraysPtr.P1[i+j*localNx+k*localNx*(def.Ny)] - P_atm)) * (1. - HostArraysPtr.S2[i+j*localNx+k*localNx*(def.Ny)]) - HostArraysPtr.roS1[i+j*localNx+k*localNx*(def.Ny)];
			F2 = ro02 * (1. + (def.beta2) * (HostArraysPtr.P1[i+j*localNx+k*localNx*(def.Ny)] + P_k - P_atm)) * HostArraysPtr.S2[i+j*localNx+k*localNx*(def.Ny)] - HostArraysPtr.roS2[i+j*localNx+k*localNx*(def.Ny)];

			PkS = AAA * def.P_d[media] / (def.lambda[media] * (1 - def.S_wr[media]));
			F1P = ro01 * (def.beta1) * (1 - HostArraysPtr.S2[i+j*localNx+k*localNx*(def.Ny)]);
			F2P = ro02 * (def.beta2) * HostArraysPtr.S2[i+j*localNx+k*localNx*(def.Ny)];
			F1S = (-1) * ro01 * (1 + (def.beta1) * (HostArraysPtr.P1[i+j*localNx+k*localNx*(def.Ny)] - P_atm));
			F2S = ro02 * (1 + (def.beta2) * (HostArraysPtr.P1[i+j*localNx+k*localNx*(def.Ny)] + P_k - P_atm + (HostArraysPtr.S2[i+j*localNx+k*localNx*(def.Ny)] * PkS)));

			det = F1P * F2S - F1S * F2P;

			HostArraysPtr.P1[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.P1[i+j*localNx+k*localNx*(def.Ny)] - (1 / det) * (F2S * F1 - F1S * F2);
			HostArraysPtr.S2[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.S2[i+j*localNx+k*localNx*(def.Ny)] - (1 / det) * (F1P * F2 - F2P * F1);
		}  
	}
}


void Border_S2(ptr_Arrays HostArraysPtr, int i, int j, int k, int localNx, int rank, int size, consts def)
{
	if ((i == 0) && ((def.NX)>2))
	{
		HostArraysPtr.S2[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.S2[i+1+j*localNx+k*localNx*(def.Ny)];
		return;
	}

	if ((i == localNx - 1) && ((def.NX)>2))
	{
		HostArraysPtr.S2[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.S2[i-1+j*localNx+k*localNx*(def.Ny)];
		return;
	}

	if ((j == (def.Ny) - 1) && ((def.Ny)>2))
	{
		HostArraysPtr.S2[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.S2[i+(j-1)*localNx+k*localNx*(def.Ny)];
		return;
	}

	if ((j==0) && ((def.Ny)>2))
	{
		int I=i_to_I(i,rank,size, def);
		if ((I>=(def.NX)/2-(def.source)) && (I<=(def.NX)/2+(def.source)) && (k>=(def.Nz)/2-(def.source)) && (k<=(def.Nz)/2+(def.source)))
			HostArraysPtr.S2[i+j*localNx+k*localNx*(def.Ny)] = S2_gr;
		else
			HostArraysPtr.S2[i+j*localNx+k*localNx*(def.Ny)] = 0;
	}

	if ((k == 0) && ((def.Nz)>2))
	{
		HostArraysPtr.S2[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.S2[i+j*localNx+(k+1)*localNx*(def.Ny)];
		return;
	}

	if ((k == (def.Nz) - 1) && ((def.Nz)>2))
	{
		HostArraysPtr.S2[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.S2[i+j*localNx+(k-1)*localNx*(def.Ny)];
		return;
	}
}

// 
void Border_P1(ptr_Arrays HostArraysPtr, int i, int j, int k, int localNx, consts def)
{
	if ((i == 0) && ((def.NX)>2))
	{
		HostArraysPtr.P1[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.P1[i+1+j*localNx+k*localNx*(def.Ny)]; 
		return;
	}

	if ((i == localNx - 1) && ((def.NX)>2))
	{
		HostArraysPtr.P1[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.P1[i-1+j*localNx+k*localNx*(def.Ny)];
		return;
	}

	if ((j == (def.Ny) - 1) && ((def.Ny)>2))
	{
		HostArraysPtr.P1[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.P1[i+(j-1)*localNx+k*localNx*(def.Ny)] + HostArraysPtr.ro1[i+localNx*1] * g_const * (def.h2); ;
		return;
	}

	if ((j==0) && ((def.Ny)>2))
	{
		HostArraysPtr.P1[i+j*localNx+k*localNx*(def.Ny)] = P_atm;
		return;
	}

	if ((k == 0) && ((def.Nz)>2))
	{
		HostArraysPtr.P1[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.P1[i+j*localNx+(k+1)*localNx*(def.Ny)]; 
		return;
	}

	if ((k == (def.Nz) - 1) && ((def.Nz)>2))
	{
		HostArraysPtr.P1[i+j*localNx+k*localNx*(def.Ny)] = HostArraysPtr.P1[i+j*localNx+(k-1)*localNx*(def.Ny)];
		return;
	}
}

// Функция загрузки данных в память хоста
void load_data_to_host(double* HostArrayPtr, double* DevArrayPtr, int localNx, consts def)
{
}

// Функция загрузки данных типа double в память ускорителя
void load_data_to_device(double* HostArrayPtr, double* DevArrayPtr, int localNx, consts def)
{
}

// Функция загрузки данных типа int в память ускорителя
void load_data_to_device_int(int* HostArrayPtr, int* DevArrayPtr, int localNx, consts def)
{
}

// Выделение памяти ускорителя под массив точек расчетной области
void device_memory_alloc(ptr_Arrays* ArraysPtr, double** DevBuffer, int nX, consts def)
{
}

// Освобожение памяти ускорителя из под массива точек расчетной области
void device_memory_free(ptr_Arrays ptDev, double* DevBuffer)
{
}

// Инициализация ускорителя
void Device_Initialize(int rank, int* blocksX, int* blocksY, int* blocksZ, int localNx, consts def)
{
}

// Загрузка на хост данных для обмена на границе
void load_exchange_data(double* HostArrayPtr, double* DevArrayPtr, double* HostBuffer, double* DevBuffer, int localNx, int blocksY, int blocksZ, int rank, int size, consts def)
{
	for(int j=0;j<(def.Ny);j++)
		for(int k=0;k<(def.Nz);k++)
		{
			HostBuffer[j+(def.Ny)*k]=HostArrayPtr[1+localNx*j+localNx*(def.Ny)*k];
			HostBuffer[j+(def.Ny)*k+(def.Ny)*(def.Nz)]=HostArrayPtr[localNx-2+localNx*j+localNx*(def.Ny)*k];
		}

	/*for(int j=0;j<(def.Ny);j++)
		for(int k=0;k<(def.Nz);k++)
			printf("Buffer j=%d k=%d buffer=%f\n", j, k, HostBuffer[j+(def.Ny)*k]);*/
}

// Загрузка на device данных обмена на границе
void save_exchange_data(double* HostArrayPtr, double* DevArrayPtr, double* HostBuffer, double* DevBuffer, int localNx, int blocksY, int blocksZ, int rank, int size, consts def)
{
	//printf("\nSave\n");
	/*for(int j=0;j<(def.Ny);j++)
		for(int k=0;k<(def.Nz);k++)
			printf("Buffer j=%d k=%d buffer=%f\n", j, k, HostBuffer[j+(def.Ny)*k]);*/

	for(int j=0;j<(def.Ny);j++)
		for(int k=0;k<(def.Nz);k++)
		{
			if(rank!=size-1)
				HostArrayPtr[localNx-1+localNx*j+localNx*(def.Ny)*k]=HostBuffer[j+(def.Ny)*k+(def.Ny)*(def.Nz)];
			if (rank!=0)
				HostArrayPtr[localNx*j+localNx*(def.Ny)*k]=HostBuffer[j+(def.Ny)*k];
		}
}