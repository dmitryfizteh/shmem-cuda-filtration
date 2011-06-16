#include <mpi.h>
#include "defines.h"

// �������� � ����� ������ ������ �������
void right_send_recv(double* HostBuffer, int destination_rank, int send_recv_id, int localNx, consts def)
{
	MPI_Status status;

	if (! MPI_Sendrecv_replace(HostBuffer+(def.Ny)*(def.Nz),(def.Ny)*(def.Nz),MPI_DOUBLE,destination_rank,send_recv_id,destination_rank,send_recv_id+1,MPI_COMM_WORLD,&status)==MPI_SUCCESS)
		printf("MPI Error: MPI_Sendrecv_replace returned an error.\n");

	//printf("Send right %d to rank=%d.\n", send_recv_id, destination_rank);

	//MPI_Send(HostBuffer+(def.Ny)*(def.Nz),(def.Ny)*(def.Nz),MPI_DOUBLE,destination_rank,send_recv_id,MPI_COMM_WORLD);
	//MPI_Recv(HostBuffer+(def.Ny)*(def.Nz),(def.Ny)*(def.Nz),MPI_DOUBLE,destination_rank,send_recv_id+1,MPI_COMM_WORLD,&status);
}

// ��������� � �������� ������ �� ����� �������
void left_recv_send(double* HostBuffer, int destination_rank, int send_recv_id, int localNx, consts def)
{
	MPI_Status status;

	if (! MPI_Sendrecv_replace(HostBuffer,(def.Ny)*(def.Nz),MPI_DOUBLE,destination_rank,send_recv_id+1,destination_rank,send_recv_id,MPI_COMM_WORLD,&status)==MPI_SUCCESS)
		printf("MPI Error: MPI_Sendrecv_replace returned an error.\n");

	//printf("Send left %d to rank=%d.\n", send_recv_id, destination_rank);

	//MPI_Recv(HostBuffer,(def.Ny)*(def.Nz),MPI_DOUBLE,destination_rank,send_recv_id,MPI_COMM_WORLD,&status);
	//MPI_Send(HostBuffer,(def.Ny)*(def.Nz),MPI_DOUBLE,destination_rank,send_recv_id+1,MPI_COMM_WORLD);
}

// ����� ������� �� �������� ����� ����� ������������
// 0. ��������� ������ � ��������� � ������ �����
// 1.  ��� ���� ������ ����������� 
// 1.1 ��������/�������� ������ �������, 
// 1.2 ��������/�������� ����� �������.
// 2.2 ��� �������� - ��������/�������� ����� �������,
// 2.2 ��������/�������� ������.
// ��� ������� ����������� ��������������� ������ �� ���������
// 3. ��������� ���������� ������ � ������ ����������
void exchange(double* HostArrayPtr, double* DevArrayPtr, double* HostBuffer, double* DevBuffer, int localNx, int blocksY, int blocksZ, int rank, int size, consts def)
{
	load_exchange_data(HostArrayPtr, DevArrayPtr, HostBuffer, DevBuffer, localNx, blocksY, blocksZ, rank, size, def); // (0)

	if(rank%2 == 0) // (1)
	{
		if (rank!=size-1)
			right_send_recv(HostBuffer, rank+1, 500, localNx, def); // (1.1)

		if (rank!=0)
			left_recv_send(HostBuffer, rank-1, 502, localNx, def); // (1.2)
	}
	else
	{
		if (rank!=0) // � ��������, ������ ��������
			left_recv_send(HostBuffer, rank-1, 500, localNx, def); // (2.1)

		if (rank!=size-1)
			right_send_recv(HostBuffer, rank+1, 502, localNx, def); // (2.2)
	}

	save_exchange_data(HostArrayPtr, DevArrayPtr, HostBuffer, DevBuffer, localNx, blocksY, blocksZ, rank, size, def); // (3)
}

// ����� ���������� ���������� �������� P2, ���������� ro1 � ro2, Xi ����� ������������
void P2_ro_Xi_exchange(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr, double* HostBuffer, double* DevBuffer, int localNx, int blocksY, int blocksZ, int rank, int size, consts def)
{
	exchange(HostArraysPtr.P2, DevArraysPtr.P2, HostBuffer, DevBuffer, localNx, blocksY, blocksZ, rank, size, def);
	exchange(HostArraysPtr.ro1, DevArraysPtr.ro1, HostBuffer, DevBuffer, localNx, blocksY, blocksZ, rank, size, def);
	exchange(HostArraysPtr.ro2, DevArraysPtr.ro2, HostBuffer, DevBuffer, localNx, blocksY, blocksZ, rank, size, def);
	exchange(HostArraysPtr.Xi1, DevArraysPtr.Xi1, HostBuffer, DevBuffer, localNx, blocksY, blocksZ, rank, size, def);
	exchange(HostArraysPtr.Xi2, DevArraysPtr.Xi2, HostBuffer, DevBuffer, localNx, blocksY, blocksZ, rank, size, def);
}


// ����� ���������� ���������� ��������� ����� ������������
// � ������ ������������� ��������� ������� ����� ������������ �� ��� X
// �������� u1y � u2y �� ���������
void u_exchange(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr, double* HostBuffer, double* DevBuffer, int localNx, int blocksY, int blocksZ, int rank, int size, consts def)
{
	exchange(HostArraysPtr.u1x, DevArraysPtr.u1x, HostBuffer, DevBuffer, localNx, blocksY, blocksZ, rank, size, def);
	exchange(HostArraysPtr.u2x, DevArraysPtr.u2x, HostBuffer, DevBuffer, localNx, blocksY, blocksZ, rank, size, def);
	//exchange(HostArraysPtr.u1y, localNx, rank, size);
	//exchange(HostArraysPtr.u2y, localNx, rank, size);
}

// ����� ���������� ���������� �������� ���� P1 � ������������ NAPL S2 ����� ������������
void P1_S2_exchange(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr, double* HostBuffer, double* DevBuffer, int localNx, int blocksY, int blocksZ, int rank, int size, consts def)
{
	exchange(HostArraysPtr.P1, DevArraysPtr.P1, HostBuffer, DevBuffer, localNx, blocksY, blocksZ, rank, size, def);



/*	for(int i=0;j<(localNx);i++)
		for(int j=0;j<(def.Ny);j++)
			for(int k=0;k<(def.Nz);k++)
				HostArraysPtr.P1[i+j*(def.Nx)+k*(def.Nx)*(def.Ny)]*/

	exchange(HostArraysPtr.S2, DevArraysPtr.S2, HostBuffer, DevBuffer, localNx, blocksY, blocksZ, rank, size, def);
}

void Communication_Initialize(int argc, char* argv[], int* size, int* rank, consts def)
{
	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,size); // The amount of processors
    MPI_Comm_rank(MPI_COMM_WORLD,rank); // The number of processor
	std::cout << "size=" <<*size<<"  "<<"rank= "<<*rank<<"\n";
}

void Communication_Finalize(void)
{
	MPI_Finalize();
}

// ���������� ������ Barrier ��� ��������� ������������
void Barrier(void)
{
	MPI_Barrier(MPI_COMM_WORLD);
}