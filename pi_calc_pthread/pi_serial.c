#include <stdio.h>
#include <stdlib.h>
int main(int argc, char *argv[]){
	long int npoints = 1000000;
	long int circle_count = 0;
	long int j = 0;

	for(j = 0; j < npoints; j++){
		double xc = rand()*1.0/RAND_MAX;
		double yc = rand()*1.0/RAND_MAX;

		if((xc*xc) + (yc*yc) < 1.0){
			circle_count = circle_count + 1;
			//printf("IN\n");
		}
		else{
			//printf("OUT\n");
		}
	}
	printf("PI=%lf\n",circle_count*4.0/npoints);
}
