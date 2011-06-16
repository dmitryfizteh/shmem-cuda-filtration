#include "defines.h"

// �������� �������
clock_t start_time, finish_time;  
// �������� ������� ��� ������ ����� ������������
double *HostBuffer;
double *DevBuffer;

int main(int argc, char* argv[])
{
	consts def;
	read_defines(argc, argv, &def);

	// ���������� ���������� ����������� � ������ �������� ����������
	int size=0, rank=0;
	// �������� ������ ������ ��������� ������� ����������
	ptr_Arrays HostArraysPtr;
	// GPU-������ ������ ��������� ������� ����������
	ptr_Arrays DevArraysPtr;
	// ������� (���������) ��������� ������� ����������
	int localNx=0, localNy=0;
	// ���������� ������ ����������
	int blocksX=0, blocksY=0, blocksZ=0;
	// ������� ����� �� �������
	int j=0;
	
	// ������������� ������������, ������� ���������� ���������� � ��������� ����������, 
	// ��������� ������, �������� ���������/����������� ������
	Initialize(&HostArraysPtr, &DevArraysPtr, &j, &localNx, &localNy, &size, &rank, &blocksX, &blocksY, &blocksZ, argc, argv, def);

	// ����
	//save_data_plots(HostArraysPtr, DevArraysPtr, 0, size, rank, localNx);
	
	start_time=clock();

	// ���� ����� �� ������� (������ �������� - ����� ���� �� �������)
	// 1. ���������� ������� P1 � S2 �� ��������� ��������� ����
	// 2. ������ (def.print_screen) ��� �� ����� ��������� ���������� � ��������� ����
	// 3. ������ save_plots ��� ������ ����������� � ������ ����� � 
	//    ����������� � ����� �������� (**), � ���� ����������� ��������� ������ (***)
	for (j++; j <= timeX/(def.dt); j++)
	{
		time_step_function(HostArraysPtr, DevArraysPtr, DevBuffer, def,j*(def.dt),localNx,localNy,rank,size,blocksX,blocksY,blocksZ); // (1)

		if ((j % (def.print_screen)) == 0) // (2)
		{
			printf ("t=%.3f\n",j*(def.dt)); 
			fflush(stdout);
		}
		if ((j % (def.save_plots)) == 0) // (3)
		{
			// ��������� 2 ������� ���������� ������ � ����� �������,
			// �.�. save ���������� ������, ����������� save_data_plots
			save_data_plots(HostArraysPtr, DevArraysPtr, j*(def.dt), size, rank, localNx, def); // (**)
			//save(HostArraysPtr, DevArraysPtr, j, rank, size, localNx); // (***)
		}
	}

	// ����� ���������� � ������� ������ ��������� � ��������
	finish_time=clock();
	finish_time-=start_time;
	printf( "Task time in seconds:\t%.2f\n", (double) finish_time/CLOCKS_PER_SEC);

	// ���������� ������ � ������������ ������
	Finalize(HostArraysPtr, DevArraysPtr, DevBuffer);

	// ��� ������� � Windows ����� ������ ��������� ��������� ���� �������
#ifdef _WIN32
	getchar();
#endif
	return 0;
}

// ������� ������� ����� �������� �� ��������� ��������� ����
// 1. ������ ��������� ��������� ro, �������� NAPL P2, ���������� Xi
// 2. ����� ����� ������������ ������������ ���������� P2, ro � Xi
// 3. ������ ��������� ���������
// 4. ����� ����� ������������ ������������ ���������� ��������� ���������
// 5. ������ ���������� roS �� ��������� ��������� ����
// 6. ������ ������� ������� �������� ���� P1 � ������������ DNAPL S2
// 7. ���������� ��������� ������� ��� P1 � S2
// 8. ����� ����� ������������ ������������ ���������� P1 � S2
void time_step_function(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr, double* DevBuffer, consts def, double t, int localNx, int localNy, int rank,int size, int blocksX, int blocksY, int blocksZ)
{
	P1_S2_exchange(HostArraysPtr, DevArraysPtr, HostBuffer, DevBuffer, localNx,blocksY, blocksZ, rank, size, def); // (8)
	ro_P2_Xi_calculation(HostArraysPtr,DevArraysPtr,def,localNx,rank,size,blocksX,blocksY, blocksZ); // (1)
	P2_ro_Xi_exchange(HostArraysPtr, DevArraysPtr, HostBuffer, DevBuffer, localNx,blocksY, blocksZ, rank, size, def); // (2)
	u_calculation(HostArraysPtr,DevArraysPtr,localNx,rank,size,blocksX,blocksY, blocksZ, def); // (3)
	u_exchange(HostArraysPtr, DevArraysPtr, HostBuffer, DevBuffer, localNx,blocksY, blocksZ, rank, size, def); // (4)
	roS_calculation(HostArraysPtr,DevArraysPtr,def,t,localNx,rank,size,blocksX,blocksY, blocksZ); // (5)
	P1_S2_calculation(HostArraysPtr,DevArraysPtr,def,localNx,rank,size,blocksX,blocksY, blocksZ); // (6)
	boundary_conditions(HostArraysPtr,DevArraysPtr,localNx,rank,size,blocksX,blocksY, blocksZ, def); // (7)
	
}

// ���������� ��������� ������ �� ���� ������
void initial_data(ptr_Arrays HostArraysPtr, int* t, int localNx, int localNy, int rank, int size, consts def)
{
	*t=0;
	for(int i=0;i<localNx;i++)
		for(int j=0;j<localNy;j++)
			for(int k=0;k<(def.Nz);k++)
				if(is_active_point(i, localNx, rank, size))
					{
						// �������������� ��������� ��������� ���������� � ����������
						int I=i_to_I(i,rank,size,def);
						// ���� ����� �� ������� �������, �� ����� (def.source) ����� �� ������,
						// �� � ��� ��������� ������������. �����, �������
						if ((j==0) && (I>=(def.NX)/2-(def.source)) && (I<=(def.NX)/2+(def.source)) && (k>=(def.Nz)/2-(def.source)) && (k<=(def.Nz)/2+(def.source)))
							HostArraysPtr.S2[i+j*localNx+k*localNx*(def.Ny)]=S2_gr;
						else
							HostArraysPtr.S2[i+j*localNx+k*localNx*(def.Ny)]=0;

						HostArraysPtr.P1[i+j*localNx+k*localNx*(def.Ny)]=P_atm+j*ro01*g_const*(def.h2);
						HostArraysPtr.x[i+j*localNx+k*localNx*(def.Ny)]=I*(def.h1);
						HostArraysPtr.y[i+j*localNx+k*localNx*(def.Ny)]=j*(def.h2);
						HostArraysPtr.z[i+j*localNx+k*localNx*(def.Ny)]=k*(def.h3);

						HostArraysPtr.media[i+j*localNx+k*localNx*(def.Ny)]=0;

					
						/*
						if ((HostArraysPtr.x[i+j*localNx+k*localNx*(def.Ny)]>=(def.NX)/2.*(def.h1)) && (HostArraysPtr.x[i+j*localNx+k*localNx*(def.Ny)]<=4.*(def.NX)/5.*(def.h1)))
							if ((HostArraysPtr.y[i+j*localNx+k*localNx*(def.Ny)]<=2./5.*(def.Ny)*(def.h2)) && (HostArraysPtr.y[i+j*localNx+k*localNx*(def.Ny)]>=(-1.)*HostArraysPtr.x[i+j*localNx+k*localNx*(def.Ny)]/4.+2./5.*(def.Ny)*(def.h2)))
								HostArraysPtr.media[i+j*localNx+k*localNx*(def.Ny)]=1;

						if ((HostArraysPtr.x[i+j*localNx+k*localNx*(def.Ny)]>=(def.NX)/5.*(def.h1)) && (HostArraysPtr.x[i+j*localNx+k*localNx*(def.Ny)]<=2.*(def.NX)/5.*(def.h1)))
							if ((HostArraysPtr.y[i+j*localNx+k*localNx*(def.Ny)]<=4./5.*(def.Ny)*(def.h2)) && (HostArraysPtr.y[i+j*localNx+k*localNx*(def.Ny)]>=3./5.*(def.Ny)*(def.h2)))
								HostArraysPtr.media[i+j*localNx+k*localNx*(def.Ny)]=1;
								*/
					
						/*
						if ((HostArraysPtr.x[i+j*localNx+k*localNx*(def.Ny)]>=2.*(def.NX)/5.*(def.h1)) && (HostArraysPtr.x[i+j*localNx+k*localNx*(def.Ny)]<=3.*(def.NX)/5.*(def.h1)))
							if ((HostArraysPtr.y[i+j*localNx+k*localNx*(def.Ny)]>=1./10.*(def.Ny)*(def.h2)) && (HostArraysPtr.y[i+j*localNx+k*localNx*(def.Ny)]<=3./10.*(def.Ny)*(def.h2)))
								HostArraysPtr.media[i+j*localNx+k*localNx*(def.Ny)]=1;
						*/
					}
}

// �������������� ��������� ��������� ���������� � ����������
// ������ ��������� �������� �������������� ����� � ������� ���
// ������ �������, ���� ����� ������ 
// (���� 2 ������ � ����� ������,�� +2 �����). 
// ���������� ������� �������� ��� ������� ����� (������ � ������� �� rank==0)
int i_to_I(int i, int rank, int size, consts def)
{
	int I;
	if (rank <= (def.NX)%size)
	{
		if(rank==0)
			I=i;
		else
			I=((def.NX)/size+1)*rank+i-1;
	}
	else
		I=((def.NX)/size+1)*rank-(rank-(def.NX)%size)+i-1;
	return I;
}

// ���������� ��������� ������� (��������) ����������
// ���� ������� �� ��������������, �� ������ (def.NX)%size �������� +1 � ��������.
// ������ ��������� �������� �������������� ����� � ������� ���
// ������ �������, ���� ����� ������ 
// (���� 2 ������ � ����� ������,�� +2 �����). 
// ���������� ������� �������� ��� ������� ����� (������ � ������� �� rank==0)
void N_to_local (int* localNx, int* localNy, int size, int rank, consts def)
{
	*localNx=(def.NX)/size;
	if (rank < (def.NX)%size)
		(*localNx)++;

	// ������� ���������� �������� �� 1 ����� ��� ��������� ������,
	// ��������� - �� 2 �� ��� �������
	// ���� ��������� ����, �� ������ � ���� ��� � �������������� ����� �� �����
	if (size>1)
	{
		if ((rank==0) || (rank==size-1))
			(*localNx)++;
		else
			(*localNx)+=2;
	}

	*localNy=(def.Ny);
	/*
	*localNy=(def.Ny)/size;
	if (rank < (def.Ny)%size)
		*localNy++;
	*/
}

// �������� �� ����� �������� (�.�. �� ��������������� ������ ��� ������ �� ��������)
int is_active_point(int i, int localNx, int rank, int size)
{
	if((rank!=0 && i==0) || (rank!=size-1 && i==localNx-1))
		return 0;
	else
		return 1;
}
//----------------------------------------------------------------------------------------------------
// ��������� �������

// ������������� ������������ (1), ������� ���������� ���������� � ��������� ���������� (2), 
// ������������� ���������� (2.5), ��������� ������ (3), �������� ���������/����������� ������ (4)
void Initialize(ptr_Arrays* HostArraysPtr, ptr_Arrays* DevArraysPtr, int* j, int* localNx, int* localNy, int* size, int* rank, int* blocksX, int* blocksY, int* blocksZ, int argc, char* argv[], consts def)
{
	FILE *f_save;

	Communication_Initialize(argc, argv, size, rank, def); // (1)

	N_to_local(localNx, localNy, *size, *rank, def); // (2)

	Device_Initialize(*rank, blocksX, blocksY, blocksZ, *localNx, def); // (2.5)

	memory_alloc(HostArraysPtr, DevArraysPtr, (def.NX)/(*size)+3, (def.Ny), def); // (3)

	// ���� ������� ��������� ����� ������� ���� ������������ ���������,
	// �� ��������������� ���������, ����� ��������� ��������� �������
	if (f_save=fopen("save/save.dat","rb"))
	{
		fclose(f_save);
		restore(*HostArraysPtr, j, *rank, *size, *localNx, def);
	}
	else
		initial_data (*HostArraysPtr, j, *localNx, *localNy, *rank, *size, def); // (4)

	load_data_to_device((*HostArraysPtr).P1, (*DevArraysPtr).P1, *localNx, def);
	load_data_to_device((*HostArraysPtr).S2, (*DevArraysPtr).S2, *localNx, def);
	load_data_to_device((*HostArraysPtr).roS1_old, (*DevArraysPtr).roS1_old, *localNx, def);
	load_data_to_device((*HostArraysPtr).roS2_old, (*DevArraysPtr).roS2_old, *localNx, def);
	load_data_to_device_int((*HostArraysPtr).media, (*DevArraysPtr).media, *localNx, def);
}

// ���������� ������ (1), ������������ ������ (2)
void Finalize(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr, double* DevBuffer)
{
	memory_free(HostArraysPtr, DevArraysPtr); // (2)
	Communication_Finalize(); // (1)
}

// ��������� ������ ����� (1) � ���������� (2) ��� ������ ����� ��������� �������
void memory_alloc(ptr_Arrays* HostArraysPtr, ptr_Arrays* DevArraysPtr, int localNx, int nY, consts def)
{
	host_memory_alloc(HostArraysPtr, localNx, nY, def); // (1)
	device_memory_alloc(DevArraysPtr, &DevBuffer, localNx, def); // (2)
}

// ����������� ������ ����� (1) � ���������� (2) �� ��� ������� ����� ��������� �������
void memory_free(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr)
{
	host_memory_free(HostArraysPtr); // (1)
	device_memory_free(DevArraysPtr, DevBuffer); // (2)
}

// ��������� ������ ����� ��� ������ ����� ��������� �������
void host_memory_alloc(ptr_Arrays* ArraysPtr, int localNx, int nY, consts def)		
{	
	if (!(HostBuffer=new double[2*(def.Ny)*(def.Nz)]))
		printf ("\nWarning! Memory for *HostBuffer is not allocated in function host_memory_alloc\n");

	try
	{
		(*ArraysPtr).x=new double [localNx*nY*(def.Nz)];
		(*ArraysPtr).y=new double [localNx*nY*(def.Nz)];
		(*ArraysPtr).z=new double [localNx*nY*(def.Nz)];
		(*ArraysPtr).P1=new double [localNx*nY*(def.Nz)];
		(*ArraysPtr).P2=new double [localNx*nY*(def.Nz)];
		(*ArraysPtr).S2=new double [localNx*nY*(def.Nz)];
		(*ArraysPtr).ro1=new double [localNx*nY*(def.Nz)];
		(*ArraysPtr).ro2=new double [localNx*nY*(def.Nz)];
		(*ArraysPtr).u1x=new double [localNx*nY*(def.Nz)];
		(*ArraysPtr).u1y=new double [localNx*nY*(def.Nz)];
		(*ArraysPtr).u1z=new double [localNx*nY*(def.Nz)];
		(*ArraysPtr).u2x=new double [localNx*nY*(def.Nz)];
		(*ArraysPtr).u2y=new double [localNx*nY*(def.Nz)];
		(*ArraysPtr).u2z=new double [localNx*nY*(def.Nz)];
		(*ArraysPtr).Xi1=new double [localNx*nY*(def.Nz)];
		(*ArraysPtr).Xi2=new double [localNx*nY*(def.Nz)];
		(*ArraysPtr).roS1=new double [localNx*nY*(def.Nz)];
		(*ArraysPtr).roS1_old=new double [localNx*nY*(def.Nz)];
		(*ArraysPtr).roS2=new double [localNx*nY*(def.Nz)];
		(*ArraysPtr).roS2_old=new double [localNx*nY*(def.Nz)];
		(*ArraysPtr).media=new int [localNx*nY*(def.Nz)];
	}
	catch(...)
	{
		printf ("\nError! Not enough host memory\n");
		exit(0);
	}
}

// ����������� ������ ����� �� ��� ������� ����� ��������� �������
void host_memory_free(ptr_Arrays ArraysPtr)
{ 
	delete HostBuffer;
	delete[] ArraysPtr.x;
	delete[] ArraysPtr.y;
	delete[] ArraysPtr.z;
	delete[] ArraysPtr.P1;
	delete[] ArraysPtr.P2;
	delete[] ArraysPtr.S2;
	delete[] ArraysPtr.ro1;
	delete[] ArraysPtr.ro2;
	delete[] ArraysPtr.u1x;
	delete[] ArraysPtr.u1y;
	delete[] ArraysPtr.u1z;
	delete[] ArraysPtr.u2x;
	delete[] ArraysPtr.u2y;
	delete[] ArraysPtr.u2z;
	delete[] ArraysPtr.Xi1;
	delete[] ArraysPtr.Xi2;
	delete[] ArraysPtr.roS1;
	delete[] ArraysPtr.roS1_old;
	delete[] ArraysPtr.roS2;
	delete[] ArraysPtr.roS2_old;
	delete[] ArraysPtr.media;
}

// ������� ���������� �������� � �����
void save_data_plots(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr, double t, int size, int rank, int localNx, consts def)
{
	// �������� � ������ ����� ����������� �������
	load_data_to_host(HostArraysPtr.P1, DevArraysPtr.P1 , localNx, def);
	load_data_to_host(HostArraysPtr.S2, DevArraysPtr.S2 , localNx, def);
	load_data_to_host(HostArraysPtr.u2x, DevArraysPtr.u2x , localNx, def);
	load_data_to_host(HostArraysPtr.u2y, DevArraysPtr.u2y , localNx, def);
	load_data_to_host(HostArraysPtr.u2z, DevArraysPtr.u2z , localNx, def);
	//load_data_to_host(HostArraysPtr.roS1, DevArraysPtr.roS1 , localNx, def);

	// �������� �� ����� �� ����������� ��������� �������� P1/P2 � S2
#ifdef TEST
	test_correct_P1_S2(HostArraysPtr, localNx, rank, def);
#endif
	
	// ������� ��������� ������� ����������, ����� � ����������� ��������� ������
	if (rank==0)
		print_plots_top (t, def);

	// �� ������� ��� ������� �� ����������� �������� ������� ������ �� ������
	// ����� ����� �������.
	for (int cpu=0; cpu<size;cpu++)
	{
		// ���������� ������ Barrier ��� ��������� ������������
		Barrier();
		if (rank==cpu)
			print_plots(HostArraysPtr, t, rank, size, localNx,def);	
	}
}

// ������� �������� ����������, ������ ��� �������� � ���������� ���������� � ��� (!3D)
void print_plots_top (double t, consts def)
{
	char fname_S2[30],fname_P1[30],fname_u[30],fname_S2y[30],fname_S2x[30];
	FILE *fp_S2,*fp_P1,*fp_u,*fp_S2y,*fp_S2x, *fp_media;

	sprintf(fname_S2,"plot_S2/S2=%012.4f.dat",t);
	sprintf(fname_P1,"plot_P1/P1=%012.4f.dat",t);
	sprintf(fname_u,"plot_u/u=%012.4f.dat",t);
	sprintf(fname_S2y,"plot_S2_y/S2=%012.4f.dat",t);
	sprintf(fname_S2x,"plot_S2_x/S2=%012.4f.dat",t);

#ifdef _WIN32
	_mkdir("plot_P1");
	_mkdir("plot_S2");
	_mkdir("plot_u");
	_mkdir("plot_S2_y");
	_mkdir("plot_S2_x");
#else
	mkdir("plot_P1",0000777);
	mkdir("plot_S2",0000777);
	mkdir("plot_u",0000777);
	mkdir("plot_S2_y",0000777);
	mkdir("plot_S2_x",0000777);
#endif

	// �������� (��� ���������� ������) � ���������
	// 1. ��� ������������� ������������� NAPL S2
	// 2. ��� ������������� �������� ���� P1
	// 3. ��� ������������� ��������� {u_x, u_y}
	// 4. ��� ����� ������������� S2 �� ��� Y
	// 5. ��� ����� ������������� S2 �� ��� X
	// 6. ��� ������������� ����� �������
	if(!(fp_S2=fopen(fname_S2,"wt")) || !(fp_P1=fopen(fname_P1,"wt")) || !(fp_u=fopen(fname_u,"wt")) || !(fp_S2y=fopen(fname_S2y,"wt")) || !(fp_S2x=fopen(fname_S2x,"wt")) || !(fp_media=fopen("media.dat","wt")))
		std::cout << "Not open file(s) in function SAVE_DATA_PLOTS! \n";

	//fprintf(fp_S2,"TITLE =  \"Saturation of DNALP in time=%5.2f\" \n", t); // (1)
	//fprintf(fp_S2,"VARIABLES = \"X\",\"Y\",\"S2\" \n");
	//fprintf(fp_S2,"ZONE T = \"BIG ZONE\", J=%d,I=%d, F = POINT\n", (def.NX), (def.Ny));

	fprintf(fp_S2,"TITLE =  \"Saturation of DNALP in time=%5.2f\" \n", t); // (1)
	fprintf(fp_S2,"VARIABLES = \"X\",\"Y\",\"Z\",\"S2\",\"P1\",\"u_x\", \"u_y\", \"u_z\" \n");
	fprintf(fp_S2,"ZONE T = \"BIG ZONE\", K=%d,J=%d,I=%d, F = POINT\n", (def.NX), (def.Nz), (def.Ny));

	fprintf(fp_P1,"TITLE =  \"Pressure of water in time=%5.2f\" \n", t); // (2)
	fprintf(fp_P1,"VARIABLES = \"X\",\"Y\",\"P1\" \n");
	fprintf(fp_P1,"ZONE T = \"BIG ZONE\", J=%d,I=%d, F = POINT\n", (def.NX), (def.Ny)); 

	fprintf(fp_u,"TITLE =  \"Velocity of DNALP in time=%5.2f\" \n", t); // (3)
	fprintf(fp_u,"VARIABLES = \"X\",\"Y\",\"u_x\", \"u_y\" \n");
	fprintf(fp_u,"ZONE T = \"BIG ZONE\", J=%d,I=%d, F = POINT\n", (def.NX)-2, (def.Ny)-2);

	fprintf(fp_S2y,"TITLE =  \"Saturation of DNALP in time=%5.2f\" \n", t); // (4)
	fprintf(fp_S2y,"VARIABLES = \"Y\",\"S2\" \n");
	fprintf(fp_S2y,"ZONE T = \"BIG ZONE\", J=%d, F = POINT\n", (def.Ny));

	fprintf(fp_S2x,"TITLE =  \"Saturation of DNALP in time=%5.2f\" \n", t); // (5)
	fprintf(fp_S2x,"VARIABLES = \"X\",\"S2\" \n");
	fprintf(fp_S2x,"ZONE T = \"BIG ZONE\", J=%d, F = POINT\n", (def.NX));

	fprintf(fp_media,"TITLE =  \"Porous media\" \n"); // (6)
	fprintf(fp_media,"VARIABLES = \"X\",\"Y\",\"media\" \n");
	fprintf(fp_media,"ZONE T = \"BIG ZONE\", J=%d,I=%d, F = POINT\n", (def.NX), (def.Ny)); 

	fclose(fp_S2);
	fclose(fp_P1);
	fclose(fp_u);
	fclose(fp_S2y);
	fclose(fp_S2x);
	fclose(fp_media);
}

// ������� ���������� ������ � ����� �������� (!3D)
void print_plots(ptr_Arrays HostArraysPtr, double t, int rank, int size, int localNx, consts def)
{
	char fname_S2[30],fname_P1[30],fname_u[30],fname_S2y[30],fname_S2x[30];
	FILE *fp_S2,*fp_P1,*fp_u,*fp_S2y,*fp_S2x, *fp_media;
	int local;

	sprintf(fname_S2,"plot_S2/S2=%012.4f.dat",t);
	sprintf(fname_P1,"plot_P1/P1=%012.4f.dat",t);
	sprintf(fname_u,"plot_u/u=%012.4f.dat",t);
	sprintf(fname_S2y,"plot_S2_y/S2=%012.4f.dat",t);
	sprintf(fname_S2x,"plot_S2_x/S2=%012.4f.dat",t);
	
	// �������� �� �������� � ���������� ��������
	// 1. ��� ������������� ������������� NAPL S2
	// 2. ��� ������������� �������� ���� P1
	// 3. ��� ������������� ��������� {u_x, u_y}
	// 4. ��� ����� ������������� S2 �� ��� Y (� ����������� ����� OX)
	// 5. ��� ����� ������������� S2 �� ��� X (� ����������� ����� OY)
	// 6. ��� ������������� ����� �������
	if(!(fp_S2=fopen(fname_S2,"at")) || !(fp_P1=fopen(fname_P1,"at")) || !(fp_u=fopen(fname_u,"at")) || !(fp_S2y=fopen(fname_S2y,"at")) || !(fp_S2x=fopen(fname_S2x,"at")) || !(fp_media=fopen("media.dat","at")))
		std::cout << "Not open file(s) in function SAVE_DATA_PLOTS! \n";

	for(int i=0; i<localNx; i++)
		for(int j=0; j<(def.Ny); j++)
			for(int k=0; k<(def.Nz); k++)
				if(is_active_point(i, localNx, rank, size))
				{
					local=i+j*localNx+k*localNx*(def.Ny);
					//fprintf(fp_S2,"%d %d %d\n", i, j, rank); // TEST
					//fprintf(fp_S2,"%.2e %.2e %.3e\n", HostArraysPtr.x[i+j*localNx+k*localNx*(def.Ny)], (def.Ny)*(def.h2)-HostArraysPtr.y[i+j*localNx+k*localNx*(def.Ny)], HostArraysPtr.S2[i+j*localNx+k*localNx*(def.Ny)]); // (1)
					fprintf(fp_S2,"%.2e %.2e %.2e %.3e %.3e %.3e %.3e %.3e\n", HostArraysPtr.x[local], HostArraysPtr.z[local], (def.Ny)*(def.h2)-HostArraysPtr.y[local], HostArraysPtr.S2[local], HostArraysPtr.P1[local], HostArraysPtr.u2x[local], HostArraysPtr.u2z[local], (-1)*HostArraysPtr.u2y[local]); // (1)
					fprintf(fp_P1,"%d %d %d %.3e %.3e %.3e %.3e %.3e\n", i, j, k, HostArraysPtr.S2[local], HostArraysPtr.P1[local], HostArraysPtr.u2x[local], HostArraysPtr.u2z[local], (-1)*HostArraysPtr.u2y[local]); // (1)
					fprintf(fp_media,"%.2e %.2e %d\n", HostArraysPtr.x[i+j*localNx+k*localNx*(def.Ny)], (def.Ny)*(def.h2)-HostArraysPtr.y[i+j*localNx+k*localNx*(def.Ny)], HostArraysPtr.media[i+j*localNx+k*localNx*(def.Ny)]);	// (6)
					fprintf(fp_u,"%.2e %.2e %.3e %.3e\n", HostArraysPtr.x[i+j*localNx+k*localNx*(def.Ny)],  (def.Ny)*(def.h2)-HostArraysPtr.y[i+j*localNx+k*localNx*(def.Ny)], HostArraysPtr.u2x[i+j*localNx+k*localNx*(def.Ny)], (-1)*HostArraysPtr.u2y[i+j*localNx+k*localNx*(def.Ny)]); // (3)
				}

	for(int i=1; i<localNx-1; i++)
		for(int j=1; j<(def.Ny)-1; j++)
			for(int k=1; k<(def.Nz)-1; k++)
				if(!((rank!=0 && i==0) || (rank!=size-1 && i==localNx-1)))
					fprintf(fp_u,"%.2e %.2e %.3e %.3e\n", HostArraysPtr.x[i+j*localNx+k*localNx*(def.Ny)],  (def.Ny)*(def.h2)-HostArraysPtr.y[i+j*localNx+k*localNx*(def.Ny)], HostArraysPtr.u2x[i+j*localNx+k*localNx*(def.Ny)], (-1)*HostArraysPtr.u2y[i+j*localNx+k*localNx*(def.Ny)]); // (3)

	// �� ����� ������, ��� ��� ������ �������, �������� �������� � �����
	for(int i=0; i<localNx; i++)
		for(int k=0; k<(def.Nz); k++)
			if ((i_to_I(i,rank,size, def)==(def.NX)/2) && is_active_point(i,localNx,rank,size))
				for(int j=0; j<(def.Ny); j++)
					fprintf(fp_S2y,"%.2e %.3e\n", HostArraysPtr.y[localNx/2+j*localNx+k*localNx*(def.Ny)], HostArraysPtr.S2[localNx/2+j*localNx+k*localNx*(def.Ny)]); // (4)
	

	for(int i=0; i<localNx; i++)
		for(int k=0; k<(def.Nz); k++)
			fprintf(fp_S2x,"%.2e %.3e\n", HostArraysPtr.x[i+localNx*(def.Ny)/2+k*localNx*(def.Ny)], HostArraysPtr.S2[i+localNx*(def.Ny)/2+k*localNx*(def.Ny)]); // (5)
		
	fclose(fp_S2);
	fclose(fp_P1);
	fclose(fp_u);
	fclose(fp_S2y);
	fclose(fp_S2x);
	fclose(fp_media);
}

// ���������� ��������� � ����
void save(ptr_Arrays HostArraysPtr, ptr_Arrays DevArraysPtr, int j, int rank, int size, int localNx, consts def)
{
	// ��������� � ������ ����� ������ �� roS_old
	// P1 � S2 ��������� ��� ��� ������� ���������� ��������,
	// x,y � media �� ���������� � �������� �� �������.
	//load_data_to_host(HostArraysPtr.P1, DevArraysPtr.P1 , localNx);
	//load_data_to_host(HostArraysPtr.S2, DevArraysPtr.S2 , localNx);
	load_data_to_host(HostArraysPtr.roS1_old, DevArraysPtr.roS1_old , localNx, def);
	load_data_to_host(HostArraysPtr.roS2_old, DevArraysPtr.roS2_old , localNx, def);

	FILE *f_save;

	if (rank==0)
	{

#ifdef _WIN32
		_mkdir("save");
#else
		mkdir("save",0000777);
#endif
	
		if(!(f_save=fopen("save/save.dat","wb")))
		{
			printf("\nError: Not open file \"save.dat\"!\n");
			exit(0);
		}
		fclose(f_save);
	}

	for (int cpu=0; cpu<size;cpu++)
	{
		// ���������� ������ Barrier ��� ��������� ������������
		Barrier();
		if (rank==cpu)
		{
			if(!(f_save=fopen("save/save.dat","ab")))
			{
				printf("\nError: Not open file \"save.dat\"!\n");
				exit(0);
			}

			fwrite(&j, sizeof(int), 1, f_save);
			fwrite(HostArraysPtr.P1, sizeof(double), localNx * (def.Ny) * (def.Nz), f_save);
			fwrite(HostArraysPtr.S2, sizeof(double), localNx * (def.Ny) * (def.Nz), f_save);
			fwrite(HostArraysPtr.x, sizeof(double), localNx * (def.Ny) * (def.Nz), f_save);
			fwrite(HostArraysPtr.y, sizeof(double), localNx * (def.Ny) * (def.Nz), f_save);
			fwrite(HostArraysPtr.z, sizeof(double), localNx * (def.Ny) * (def.Nz), f_save);
			fwrite(HostArraysPtr.roS1_old, sizeof(double), localNx * (def.Ny) * (def.Nz), f_save);
			fwrite(HostArraysPtr.roS2_old, sizeof(double), localNx * (def.Ny) * (def.Nz), f_save);
			fwrite(HostArraysPtr.media, sizeof(int), localNx * (def.Ny) * (def.Nz), f_save);
			fclose(f_save);
		}
	}
}

// �������������� ��������� �� �����
void restore (ptr_Arrays HostArraysPtr, int* j, int rank, int size, int localNx, consts def)
{
	FILE *f_save;
	for (int cpu=0; cpu<size;cpu++)
	{
		// ���������� ������ Barrier ��� ��������� ������������
		Barrier();
		int lNx=0, lNy=0;
		if (rank==cpu)
		{
			if(!(f_save=fopen("save/save.dat","rb")))
			{
					printf("\nError: Not open file \"save.dat\"!\n");
					exit(0);
			}
			for (int queue=0;queue<=rank;queue++)
			{
				N_to_local(&lNx,&lNy,size,queue, def);
				fread(j, sizeof(int), 1, f_save);
				fread(HostArraysPtr.P1, sizeof(double), lNx * (def.Ny) * (def.Nz), f_save);
				fread(HostArraysPtr.S2, sizeof(double), lNx * (def.Ny) * (def.Nz), f_save);
				fread(HostArraysPtr.x, sizeof(double), lNx * (def.Ny) * (def.Nz), f_save);
				fread(HostArraysPtr.y, sizeof(double), lNx * (def.Ny) * (def.Nz), f_save);
				fread(HostArraysPtr.z, sizeof(double), lNx * (def.Ny) * (def.Nz), f_save);
				fread(HostArraysPtr.roS1_old, sizeof(double), lNx * (def.Ny) * (def.Nz), f_save);
				fread(HostArraysPtr.roS2_old, sizeof(double), lNx * (def.Ny) * (def.Nz), f_save);
				fread(HostArraysPtr.media, sizeof(int), lNx * (def.Ny) * (def.Nz), f_save);
			}
			fclose(f_save);
		}
	}
}



//------------------------------------------------------------------------------------------
// ������������

// ������� �������� �� ����� �� ����������� ��������� �������� P1/P2 � S2
// �� ���� ������ ��������� ������� ����������
void test_correct_P1_S2(ptr_Arrays HostArraysPtr, int localNx, int rank, consts def)
{
	for(int i=0;i<localNx;i++)
		for(int j=0;j<(def.Ny);j++)
			for(int k=0;k<(def.Nz);k++)
			{
				if (HostArraysPtr.S2[i+j*localNx+k*localNx*(def.Ny)]<0)
					printf ("\nWarning! S2<0 in point i=%d, j=%d, k=%d, rank=%d\n",i,j,k,rank);
				if (HostArraysPtr.P1[i+j*localNx+k*localNx*(def.Ny)]<=0)
					printf ("\nWarning! P<=0 in point i=%d, j=%d, k=%d, rank=%d\n",i,j,k,rank);
			}
}

// ������ unit-������
void Unit_tests(void)
{

}

// ���������� ���������� ������ �� �����
void read_defines(int argc, char *argv[], consts* def)
{
	(*def).K[0]=K[0];
	(*def).K[1]=K[1];
	(*def).lambda[0]=lambda[0];
	(*def).lambda[1]=lambda[1];
	(*def).S_wr[0]=S_wr[0];
	(*def).S_wr[1]=S_wr[1];
	(*def).m[0]=m[0];
	(*def).m[1]=m[1];
	(*def).P_d[0]=P_d[0];
	(*def).P_d[1]=P_d[1];

	FILE *defs;
	char *file;
	char str[250]="", attr_name[50]="", attr_value[50]="";

	file=DEFINES_FILE;

	if(!(defs=fopen(file,"rt")))
	{
		printf("Not open file \"%s\"!\nError in file \"%s\" at line %d\n", file,__FILE__,__LINE__);
		fflush(stdout);
	}
	else
	{
		while (!feof(defs))
		{
			unsigned int i,j,a;
			fgets(str,250,defs);
			if(str[0]=='#')
				continue;
			for (i=0;str[i]!='=';i++)
			{
				if (i>=strlen(str))
					continue;
				attr_name[i]=str[i];
			}

			attr_name[i]='\0';
			a=strlen(str);
			for(j=i+1;str[j]!=' ' && (j < a);j++)
				attr_value[j-i-1]=str[j];
			attr_value[j-i-1]='\0';


			if(!strcmp(attr_name,"H1")) 
				(*def).h1 = atof(attr_value);
			if(!strcmp(attr_name,"H2")) 
				(*def).h2 = atof(attr_value);
			if(!strcmp(attr_name,"H3")) 
				(*def).h3 = atof(attr_value);
			if(!strcmp(attr_name,"TAU")) 
				(*def).tau = atof(attr_value);
			if(!strcmp(attr_name,"DT")) 
				(*def).dt = atof(attr_value);
			if(!strcmp(attr_name,"L_W")) 
				(*def).l_w = atof(attr_value);
			if(!strcmp(attr_name,"L_N")) 
				(*def).l_n = atof(attr_value);
			if(!strcmp(attr_name,"C")) 
				(*def).c = atof(attr_value);
			if(!strcmp(attr_name,"BETA1")) 
				(*def).beta1 = atof(attr_value);
			if(!strcmp(attr_name,"BETA2")) 
				(*def).beta2 = atof(attr_value);

			if(!strcmp(attr_name,"SOURCE"))
				(*def).source = atoi(attr_value);
			if(!strcmp(attr_name,"SAVE_PLOTS"))
				(*def).save_plots = atoi(attr_value);
			if(!strcmp(attr_name,"PRINT_SCREEN"))
				(*def).print_screen = atoi(attr_value);
			if(!strcmp(attr_name,"NX"))
				(*def).NX = atoi(attr_value);
			if(!strcmp(attr_name,"NY"))
				(*def).Ny = atoi(attr_value);
			if(!strcmp(attr_name,"NZ"))
				(*def).Nz = atoi(attr_value);
		}

		fclose(defs);
	}
}