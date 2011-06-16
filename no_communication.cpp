#include "defines.h"

void P2_ro_Xi_exchange(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr, double* HostBuffer, double* DevBuffer, int localNx, int blocksY, int blocksZ, int rank, int size, consts def)
{
	//load_exchange_data(HostArraysPtr.P1, DevArraysPtr.P1, HostBuffer, DevBuffer, localNx, blocksY, blocksZ, rank, size, def); // TEST
}

void u_exchange(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr, double* HostBuffer, double* DevBuffer, int localNx, int blocksY, int blocksZ, int rank, int size, consts def)
{
}

void P1_S2_exchange(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr, double* HostBuffer, double* DevBuffer, int localNx, int blocksY, int blocksZ, int rank, int size, consts def)
{
}

void Communication_Initialize(int argc, char* argv[], int* size, int* rank, consts def)
{
	*size=1;
	*rank=0;
}

void Communication_Finalize(void)
{
}

// Реализация фунции Barrier для различных коммуникаций
void Barrier(void)
{
}