#include <stdio.h>
#include <time.h> 
#include <cuda.h>
#include <malloc.h>
#include <iostream>
#include <fstream>
#include <string>
#include <string.h>
using namespace std;

// #define DEBUG

// the first 3 bits of a pointer are empty, use the first bit as marker
#define IS_MARKED(p)  ((int)(((unsigned long long)(p)) & 1))
#define GET_MARKED_REF(p) (((unsigned long long)(p)) | 1)
#define GET_UNMARKED_REF(p) (((unsigned long long)(p)) & ~1)

struct node {
	int data;
	struct node *next;
};

__device__ struct node *head;
__device__ struct node *tail;

__device__ struct node *createNode(int val) {
	struct node *newnode = (struct node *)malloc(sizeof(struct node));
	newnode->data = val;
	newnode->next = NULL;
	return newnode;
}

__global__ void listInit()
{
	head = createNode(-1);
	tail = createNode(-2);
	head->next=tail;
}

__device__ void addFront(struct node *newnode)
{
	newnode->next = head->next;
	head->next = newnode;
}

__global__ void addFront(int val)
{
	// need to modify
	struct node *newnode=createNode(val);
	addFront(newnode);
}

__global__ void addFront(int *arr, int N)
{
	for(int i=0;i<N;i++)
	{
		struct node *newnode=createNode(arr[i]);
		addFront(newnode);
	}
}

__device__ void nodePrint(struct node *ptr) {
	if (ptr==head)
		printf("head ");
	else if (ptr==tail)
		printf("tail\n");
	else
		printf("%d ", ptr->data);
}

__global__ void listPrint() {
	printf("listPrint\n");
	int nnodes = 0;
	for (struct node *ptr = head; ptr; ptr = (struct node *)GET_UNMARKED_REF(ptr->next), ++nnodes)
	{
		nodePrint(ptr);
	}
	printf("Number of nodes = %d\n", nnodes);
}

__global__ void listPrintLen() {
	int nnodes = 0;
	for (struct node *ptr = head; ptr; ptr = (struct node *)GET_UNMARKED_REF(ptr->next), ++nnodes)
	{
	}
	printf("Number of nodes = %d\n", nnodes);
}

__global__ void listPrintRaw() {
	// print with marked nodes
	printf("listPrintRaw\n");
	int nnodes = 0;
	for (struct node *ptr = head; ptr; ptr = (struct node *)GET_UNMARKED_REF(ptr->next))
	{	
		if (ptr == tail)
		{
			nodePrint(ptr);
			nnodes++;
		}
		else if(!IS_MARKED(ptr->next))
		{
			nodePrint(ptr);
			nnodes++;
		}
	}
	printf("Number of nodes = %d\n", nnodes);
}

__device__ void printVal(int val)
{
	printf("val: %d\n",val);
}

__global__ void printVal(int *arr, int N)
{
	for (int i=0;i<N;i++)
	{
		printVal(arr[i]);
	}
}

__device__ struct node *listSearch(int val, int &collision)
{
#ifdef DEBUG
	printf("listSearch val: %d\n", val);
#endif
	struct node *cur=NULL, *p, *prev_next;
	struct node *prev;
	int cnt = 0;
	while(1)
	{
		// step1: traverse the list and find the node
		for(cur=head; cur->next; cur=(struct node *)GET_UNMARKED_REF(cur->next))
		{
			if (cur==tail)
			{
				break;  // not found
			}
			if (IS_MARKED(cur->next))  // p->next is marked means p is deleted logically
			{
#ifdef DEBUG
				printf("next is marked\n");
#endif
				continue;  // skip this node
			}
			if (cur->data == val)  // found
			{
				break;
			}
			prev=cur;

		}
#ifdef DEBUG
		if(cur==tail)  // cur is the tail node
		{
			printf("listSearch %d not found, prev data: %d, prev->next data: %d\n", val, prev->data, prev->next->data);
		}
		else
			printf("listSearch val found, cur->data: %d, cur ref: %llu\n", cur->data, GET_UNMARKED_REF(cur));
#endif
		// no marked nodes between prev and cur
		if (prev->next == cur)
		{
			if (cur==tail)  // cur not found, cur is tail node
			{
#ifdef DEBUG
				printf("cur reaches the tail\n");
#endif
				break;  // then return cur
			}
			else if (!IS_MARKED(cur->next))  // if cur is marked as removed during the time, search again
			{
				// collision++;
				break;  // then return cur
			}
		}
		
		// step2: remove marked nodes between prev and cur
		else
		{
#ifdef DEBUG
			printf("prev data: %d, prev->next data: %d, cur data: %d\n", prev->data, (prev->next)->data, cur->data);
#endif 
			// Step 2.1: If an insertions was made in the meantime between left and right, repeat search.
			int inserted = 0;
			for(p=(struct node *)GET_UNMARKED_REF(prev->next); p==cur; p=(struct node *)GET_UNMARKED_REF(p->next))
			{
				// loop from prev to cur, if there is any unmarked node, it is inserted meantime, need to search again
                if (!IS_MARKED(p->next))
					inserted = 1;
			}
			if (inserted==1)
			{
				collision++;
				continue;  // search again
			}
			
			// No unmarked nodes in between now
			// Step 2.2: Try to "remove" the marked nodes between left and right.
			prev_next = (struct node *)atomicCAS((unsigned long long *)&prev->next, GET_UNMARKED_REF(prev->next), (unsigned long long)cur);  
			// update prev->next to cur, delete marked nodes in between (no garbage collection yet)
            if(prev_next!=(struct node *)GET_UNMARKED_REF(prev->next))
			{
#ifdef DEBUG
				if(!prev_next) printf("prev_next NULL\n");
				else printf("prev_next->data: %d\n",prev_next->data);
				if(!prev->next) printf("prev->next NULL\n");
#endif
				// somone changed left->next, deletion failed, search again
				continue;
			}
        }
	}
	if (cur==tail) cur = NULL;
	return cur;
}

__device__ void listTraverseDel()
{
	struct node *cur, *prev, *p, *prev_next;
	prev=head;
	cur=head->next;
	for(cur=head->next; cur->next; cur=(struct node*)GET_UNMARKED_REF(cur->next))
	{
		if(IS_MARKED(cur->next))  // p->next is marked means p is deleted logically
		{
			continue;  // skip this node
		}
		if(prev->next!=cur)  // stop here and do deletion
		{
#ifdef DEBUG
			printf("listTraversalDel delete: %d between prev: %d and cur: %d\n", prev->next->data, prev->data, cur->data);
#endif
			prev->next=cur;
		}
		prev=cur;
	}
}

__global__ void listTraverse()
{
	// delete marked nodes during traversal
	listTraverseDel();
}

__device__ int listInsert (int insertPrev, int insertVal) {
	int collision = 0;
	struct node *myold, *actualold;
	struct node *prev = listSearch(insertPrev, collision);
	if (prev)
	{
#ifdef DEBUG
		printf("Insert Prev %d found %d\n", insertPrev, prev->data);
#endif
		struct node *newnode = createNode(insertVal);

		do {
			myold = prev->next;  // should reload every iteration
			newnode->next = myold;
			actualold = (struct node *)atomicCAS((unsigned long long *)&prev->next, (unsigned long long)myold, (unsigned long long)newnode);  
			// collision++;
		} while (actualold != myold);
#ifdef DEBUG
		printf("Insert %d after %d\n", newnode->data, prev->data);
#endif
	}
#ifdef DEBUG
	else printf("Insert Prev %d not found\n", insertPrev);
#endif
	return collision;
}

__device__ int listRemove(int rmVal)
{
	int collision = 0;
	int val = rmVal;
	struct node *prev, *cur, *succ, *actual_succ;
	prev = cur = succ = NULL;
	int cnt=0;

	while(1)
	{
		cur = listSearch(val, collision);
		// cur = listSearch(val, &prev);
		if (cur==NULL || cur->data != val)
		{
#ifdef DEBUG
			printf("Remove node %d not found\n", val);
#endif
			break;
		}
		else
		{
#ifdef DEBUG
			printf("Remove node %d found %d\n", val, cur->data);
#endif
			succ = cur->next;
			if(!IS_MARKED(succ))
			{
				actual_succ = (struct node *)atomicCAS((unsigned long long *)&cur->next, (unsigned long long)succ, GET_MARKED_REF(succ));  // actual cur->next set as marked succ
				if(actual_succ==succ)
				{
#ifdef DEBUG
					printf("Remove found %d\n", val);
					printf("unmarked succ: %llu, marked succ: %llu, succ: %llu, actual succ: %llu, cur->next: %llu\n", GET_UNMARKED_REF(succ), GET_MARKED_REF(succ), (unsigned long long)succ, (unsigned long long)actual_succ, (unsigned long long)cur->next);
#endif
					break;
				}
				else collision++;
			}
		}
	}
	return collision;

	// try to delete physically, envoke search(succ)
	// slows down too much, use listTraverseDel instead. good for demo.
/*
	struct node *actual_prev_next = (struct node*)atomicCAS((unsigned long long *)&prev->next, (unsigned long long)cur, GET_UNMARKED_REF(succ));
	if (actual_prev_next!=cur)
	{
#ifdef DEBUG
		printf("Physically delete in remove, call listSearch\n");
#endif
		// listSearch(succ->data, &cur);
	}
#ifdef DEBUG
	else printf("Physically delete in remove\n");
#endif
*/
}

__global__ void kernel(int *ops, int *opNodes, int *opVals, int N, int *collisions) {
	// need to iteratively process all N. one threads need to process more than once
	int i=0;
	int idx =  i * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x; 
	while (idx < N)
	{
#ifdef DEBUG
		printf("idx: %d\n", idx);
#endif
		if (idx<N && ops[idx]==1)  // insert
		{
			collisions[idx]=listInsert(opNodes[idx], opVals[idx]);
#ifdef DEBUG
			printf("[insert] collision at thread %d: %d\n", idx, collisions[idx]);
#endif
		}
		else if (idx<N && ops[idx]==0)  // delete
		{
			collisions[idx]=listRemove(opNodes[idx]);
#ifdef DEBUG
			printf("[delete] collision at thread %d: %d\n", idx, collisions[idx]);
#endif
		}
		i++;
		idx = i * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x; 
#ifdef DEBUG
		printf("next idx: %d\n", idx);
#endif
	}
}

void Demo() {
	// init list
	printf("listInit\n");
	listInit<<<1,1>>>();
	addFront<<<1,1>>>(3);
	addFront<<<1,1>>>(2);
	addFront<<<1,1>>>(1);
	listPrint<<<1, 1>>>();
	cudaDeviceSynchronize();

	// generate operation data
	int ops_h[8] = {1,1,1,1,1,0,0,0};
	int *ops;
	cudaMalloc((void**)&ops, sizeof(int)*8);
	cudaMemcpy(ops, ops_h, sizeof(int)*8, cudaMemcpyHostToDevice);

	int *opVal_h = (int *)malloc(sizeof(int)*8);
	opVal_h[0]=50;
	opVal_h[1]=60;
	opVal_h[2]=70;
	opVal_h[3]=80;
	opVal_h[4]=90;
	opVal_h[5]=1;
	opVal_h[6]=80;
	opVal_h[7]=70;

	int *opVal_d;
	cudaMalloc((void **)&opVal_d, sizeof(int)*8);
	cudaMemcpy(opVal_d, opVal_h, sizeof(int)*8, cudaMemcpyHostToDevice);

	int *opNode_h = (int *)malloc(sizeof(int)*8);
	opNode_h[0]=2;
	opNode_h[1]=2;
	opNode_h[2]=2;
	opNode_h[3]=1;
	opNode_h[4]=3;
	opNode_h[5]=0;
	opNode_h[6]=0;
	opNode_h[7]=0;

	int *opNode_d;
	cudaMalloc((void **)&opNode_d, sizeof(int)*8); 
	cudaMemcpy(opNode_d, opNode_h, sizeof(int)*8, cudaMemcpyHostToDevice);

	// collision
	int *collision_d;
	cudaMalloc((void**)&collision_d, sizeof(int)*8);

	int n_blocks = 1, n_threads = 1;
	kernel<<<n_blocks, n_threads>>>(ops, opNode_d, opVal_d, 8, collision_d);
	cudaDeviceSynchronize();  // necessary!
	listPrintRaw<<<1, 1>>>();
	cudaDeviceSynchronize();
	listPrint<<<1, 1>>>();
	cudaDeviceSynchronize();
	
	// traverse to delete the marked nodes
	printf("\nlistTraverse\n");
	listTraverse<<<1,1>>>();
	cudaDeviceSynchronize();
	listPrintRaw<<<1, 1>>>();
	cudaDeviceSynchronize();
	listPrint<<<1, 1>>>();
	cudaDeviceSynchronize();

	// print collision
	int collision_h[8] = {0,0,0,0,0,0,0,0};
	cudaMemcpy(collision_h, collision_d, sizeof(int)*8, cudaMemcpyDeviceToHost);
	int sum=0;
	for (int i=0;i<8;i++)
	{
		sum+=collision_h[i];
#ifdef DEBUG
		printf("collision at thread %d: %d\n", i, collision_h[i]);
#endif
	}
	printf("total atomic collisions: %d\n",sum);
}

void readFileNodes(int *&Nodes, int &N)
{
    ifstream f;
    f.open("../data/listnodes.txt");
    if(f.is_open())
    {
        f >> N;
        printf("Read From File Nodes: %d\n",N);
        Nodes = (int *)malloc(sizeof(int)*N);
        int i=0;
        while(!f.eof())
        {
            f >> Nodes[i];
            i++;
        }
    }
    f.close();
}

void readFileOps(int *&ops, int *&opNodes, int *&opVals, int &N)
{
    ifstream f;
    f.open("../data/operations.txt");
    if(f.is_open())
    {
        f >> N;
        printf("Read From File Operations: %d\n",N);
        ops = (int *)malloc(sizeof(int)*N);
        opNodes = (int *)malloc(sizeof(int)*N);
        opVals = (int *)malloc(sizeof(int)*N);
        memset(opVals, 0, sizeof(int)*N);
        int i=0;
        while(!f.eof())
        {
            f >> ops[i];
            f >> opNodes[i];
            if(ops[i]==1)
            {
                f >> opVals[i];
            }
			else
			{
				opVals[i] = -99;  // delete operation does not have opVal
			}
            i++;
        }
    }
    f.close();
}

void parallelOperate(int n_blocks, int n_threads)
{
	struct timespec start, end;
	double time_taken;

	// read node file
	int *Nodes_h, N;
	readFileNodes(Nodes_h, N);
	// read operation file
	int *ops_h, *opNodes_h, *opVals_h, opN;
	readFileOps(ops_h, opNodes_h, opVals_h, opN);
	
	// move data to gpu
	int *Nodes, *ops, *opNodes, *opVals;
	cudaMalloc((void**)&Nodes, sizeof(int)*N);
	cudaMalloc((void**)&ops, sizeof(int)*opN);
	cudaMalloc((void**)&opNodes, sizeof(int)*opN);
	cudaMalloc((void**)&opVals, sizeof(int)*opN);
	cudaMemcpy(Nodes,Nodes_h,sizeof(int)*N,cudaMemcpyHostToDevice);
	cudaMemcpy(ops,ops_h,sizeof(int)*opN,cudaMemcpyHostToDevice);
	cudaMemcpy(opNodes,opNodes_h,sizeof(int)*opN,cudaMemcpyHostToDevice);
	cudaMemcpy(opVals,opVals_h,sizeof(int)*opN,cudaMemcpyHostToDevice);

	// collision count
	int *collision_d;
	cudaMalloc((void**)&collision_d, sizeof(int)*opN);

	// init list
	listInit<<<1,1>>>();
	addFront<<<1,1>>>(Nodes, N);
	listPrintLen<<<1, 1>>>();
	cudaDeviceSynchronize();

#ifdef DEBUG
	listPrint<<<1, 1>>>();
	cudaDeviceSynchronize();
#endif

	// parallel insert and remove operations
	clock_gettime(CLOCK_REALTIME, &start);

	kernel<<<n_blocks, n_threads>>>(ops, opNodes, opVals, opN, collision_d);
	cudaDeviceSynchronize();
  	// traverse to physically delete the marked nodes (no garbage collection yet)
  	listTraverse<<<1,1>>>();
	cudaDeviceSynchronize();
	
	clock_gettime(CLOCK_REALTIME, &end);
  	time_taken = ( end.tv_sec - start.tv_sec ) * 1000000000.0 + ( end.tv_nsec - start.tv_nsec );
	
	listPrintLen<<<1, 1>>>();
	cudaDeviceSynchronize();

#ifdef DEBUG
	listPrint<<<1, 1>>>();
	cudaDeviceSynchronize();
#endif

  	printf("Time taken = %lf msec\n", time_taken/1000000.0);

	// print collision
	int *collision_h = (int *)malloc(sizeof(int)*opN);
	cudaMemcpy(collision_h, collision_d, sizeof(int)*opN, cudaMemcpyDeviceToHost);
	int sum=0;
	for (int i=0;i<opN;i++)
	{
		sum+=collision_h[i];
#ifdef DEBUG
		printf("collision at thread %d: %d\n", i, collision_h[i]);
#endif
	}
	printf("total atomic collisions: %d\n",sum);
}

int main(int argc, char *argv[])
{
	if (argc==2)
	{
		Demo();
	}
	else if (argc==3)
	{
		int n_blocks = atoi(argv[1]);
		int n_threads = atoi(argv[2]);
		parallelOperate(n_blocks, n_threads);
	}
	else
	{
		printf("need arguments\n");
	}
	return 0;
}

/* Demo output

listInit
listPrint
head 1 2 3 tail
Number of nodes = 5

listInsert
listSearch val: 2
listSearch val: 2
listSearch val: 2
listSearch val: 1
listSearch val: 3
val found, cur->data: 2, cur ref: 140077894203920
val found, cur->data: 2, cur ref: 140077894203920
val found, cur->data: 2, cur ref: 140077894203920
val found, cur->data: 1, cur ref: 140077894204000
val found, cur->data: 3, cur ref: 140077894203840
listPrint
head 1 80 2 70 60 50 3 90 tail
Number of nodes = 10

listRemove
listSearch val: 1
listSearch val: 80
listSearch val: 70
val found, cur->data: 1, cur ref: 140077894204000
val found, cur->data: 80, cur ref: 140077894204400
val found, cur->data: 70, cur ref: 140077894204320
listPrintRaw
head 2 60 50 3 90 tail
Number of nodes = 7
listPrint
head 1 80 2 70 60 50 3 90 tail
Number of nodes = 10

listTraverse
prev: -1, cur: 2
prev: 2, cur: 60
listPrintRaw
head 2 60 50 3 90 tail
Number of nodes = 7
listPrint
head 2 60 50 3 90 tail
Number of nodes = 7

*/