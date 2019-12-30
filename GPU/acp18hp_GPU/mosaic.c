// mosaic.c : Defines the entry point for the console application.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define FAILURE 0
#define SUCCESS !FAILURE
#define USER_NAME "acp18hp"        //replace with your user name
#define RGB_COLOR 255

typedef enum MODE { CPU, OPENMP, CUDA, ALL } MODE;
typedef struct {
	unsigned char r, g, b;
	//int r, g, b;
} Pixel;
typedef struct {
	char* type; //P3 or P6
	int width, height;
	Pixel *data;
	int avg_r, avg_g, avg_b;
} image;

image* img;
unsigned int c = 0;
MODE execution_mode = CPU;
char type[3];

void print_help();
int process_command_line(int argc, char *argv[]);
image * readFile(const char *filename);
image * do_mosaic(image* img, int c);
void writeFile(const char *filename, image* img);
static image * readFile(const char *inputname) {
	//char buff[];
	image *img;    //return value
				   //alloc memory form img
	img = (image*)malloc(sizeof(image));
	FILE *fp;
	int c, rgb_color;
	//open both P3, P6 PPM file for reading it in rb
	fp = fopen(inputname, "rb");
	if (!fp) {
		fprintf(stderr, "Unable to open file '%s'\n", inputname);
		exit(1);
	}

	fscanf(fp, "%s", type);
	img->type = type;
	while (fgetc(fp) != '\n');
	//ignore comments
	c = fgetc(fp); //read '#'
	while (c == '#') {
		while (fgetc(fp) != '\n');
		c = getc(fp);
	}
	//re-read the character written lastly
	ungetc(c, fp);
	//read img size information -
	if (fscanf(fp, "%d\n%d", &img->width, &img->height) != 2) {
		//printf("%d %d", img->width, img->height);
		fprintf(stderr, "Invalid img size (error loading '%s')\n", inputname);
		exit(1);
	}

	//read rgb component
	if (fscanf(fp, "%d", &rgb_color) != 1) {
		fprintf(stderr, "Invalid rgb component (error loading '%s')\n", inputname);
		exit(1);
	}

	//check rgb component depth
	if (rgb_color != RGB_COLOR) {
		fprintf(stderr, "'%s' does not have 8-bits components\n", inputname);
		exit(1);
	}

	while (fgetc(fp) != '\n');
	//memory allocation for pixel data
	img->data = (Pixel*)malloc(img->width * img->height * sizeof(Pixel));

	if (strcmp(img->type, "P3") == 0) { //PlainText
										//printf("P3 in readFile()");

										//read pixel data from file
		int i;
		for (i = 0; i < img->width*img->height; i++) {
			if (fp) {
				fscanf(fp, "%d %d %d", &img->data[i].r, &img->data[i].g, &img->data[i].b);
				//printf("%d pixel : (%d %d %d)\n", i, img->data[i].r, img->data[i].g, img->data[i].b);
			}
		}
	}
	else { //P6
		   //printf("P6 in readFile()");

		   //read pixel data from file
		if (fread(img->data, 3 * img->width, img->height, fp) != img->height) {
			fprintf(stderr, "Error loading img '%s'\n", inputname);
			exit(1);
		}
	}
	fclose(fp);

	//After reading, debugging codes and check the metadata of image
	//printf("[After readFile()]\n");
	//printf("type : %s\n", img->type);
	//printf("size : %d * %d\n", img->width, img->height);

	return img;
}
//CPU mode
static image * do_mosaic(image* img, int c) {

	//calculate avg
	int i, j, k, l;
	int f_idx, c_idx;
	int s_r, s_g, s_b; //the sum of pixel values of the img
	int sum_r, sum_g, sum_b; //the sum of all pixels in each mosaic cell
	int avg_r, avg_g, avg_b; //the avg of each mosaic cell

							 //initialization
	s_r = 0; s_g = 0; s_b = 0;
	// i : the number of output cells
	for (i = 0; i < img->height; i += c) {
		// j : the row(height)
		for (j = 0; j < img->width; j += c) {
			// the index of the first pixel in each mosaic cell
			f_idx = img->width * i + j;

			sum_r = 0; sum_g = 0; sum_b = 0;
			avg_r = 0; avg_g = 0; avg_b = 0;

			//Pixel pix = img->data[f_idx];
			//printf("%d th pixel : ( %d %d %d ) \n", f_idx, pix.r, pix.g, pix.b);
			int cnt_per_cell = 0;
			for (k = 0; k < c; k++) {
				for (l = 0; l < c; l++) {
					//current index
					c_idx = f_idx + img->width * k + l;
					Pixel pix = img->data[c_idx];
					//printf("%d th pixel : ( %d %d %d ) \n", c_idx, img->data[c_idx].r, img->data[c_idx].g, img->data[c_idx].b);
					//printf("%d th pixel\n", c_idx);
					sum_r += pix.r;
					sum_g += pix.g;
					sum_b += pix.b;

					cnt_per_cell++;
				}
			}//finish summation of each mosaic cell

			 //summation of all pixels in the img
			s_r += sum_r;
			s_g += sum_g;
			s_b += sum_b;

			avg_r = sum_r / cnt_per_cell;
			avg_g = sum_g / cnt_per_cell;
			avg_b = sum_b / cnt_per_cell;

			//printf("After averaging\n");

			for (k = 0; k < c; k++) {
				for (l = 0; l < c; l++) {
					//current index
					c_idx = f_idx + img->width * k + l;
					//Pixel pix = img->data[c_idx];
					//printf("%d th pixel : ( %d %d %d ) \n", c_idx, pix.r, pix.g, pix.b);

					img->data[c_idx].r = avg_r;
					img->data[c_idx].g = avg_g;
					img->data[c_idx].b = avg_b;
					//check
					//printf("%d th pixel : ( %d %d %d ) \n", c_idx, img->data[c_idx].r, img->data[c_idx].g, img->data[c_idx].b);
				}
			}//finish summation of each mosaic cell
			 //printf("\n");
		}
	}
	//the avg pixel values of the img
	int pixels = img->width*img->height;
	img->avg_r = s_r / pixels;
	img->avg_g = s_g / pixels;
	img->avg_b = s_b / pixels;
	//printf("%d %d %d", img->avg_r, img->avg_g, img->avg_b);

	return img;
}
//OPENMP mode
static image * do_mosaic_omp(image* img, int c) {

	//calculate avg
	int i = 0, j = 0, k = 0, l = 0;
	int f_idx, c_idx;
	int s_r, s_g, s_b; //the sum of pixel values of the img
	int sum_r, sum_g, sum_b; //the sum of all pixels in each mosaic cell
	int avg_r, avg_g, avg_b; //the avg of each mosaic cell
	s_r = 0; s_g = 0; s_b = 0;


	//printf("This system has %d CPU, and now %d CPU are available.\n", max_procs);

	// firstprivate(i,j,k,l) private(sum_r,sum_g,sum_b, avg_r,avg_g,avg_b,  s_r,s_g,s_b, cnt_per_cell, f_idx, c_idx )

	//int id = omp_get_thread_num();
	//int max = omp_get_max_threads();
	//omp_set_num_threads(max);
	//printf("omp max thread %d\n", num); 
	//extern int parallelism_enabled;
	//#pragma omp parallel for collapse(4) private(i,j,f_idx, cnt_per_cell) shared(s_r, s_g, s_b) schedule(dynamic)
	// i : the number of output cells
#pragma omp parallel for private(i,j,k,l, sum_r,sum_g,sum_b, avg_r,avg_g,avg_b, f_idx, c_idx) schedule(guided)
	for (i = 0; i < img->height; i += c) {
		// j : the row(height)
		//#pragma omp for reduction(+:s_r, +:s_g, +:s_b)
		for (j = 0; j < img->width; j += c) {
			// the index of the first pixel in each mosaic cell
			f_idx = img->width * i + j;

			sum_r = 0; sum_g = 0; sum_b = 0;
			avg_r = 0; avg_g = 0; avg_b = 0;

			int cnt_per_cell = 0;

			//#pragma omp for collapse(2)
			//#pragma omp parallel for
			//{
			//#pragma omp for private(k, l, c_idx, sum_r, sum_g, sum_b)
			for (k = 0; k < c; k++) {
				for (l = 0; l < c; l++) {

					//printf("Thread number : %d\n", omp_get_thread_num());
					//current index
					c_idx = f_idx + img->width * k + l;
					Pixel pix = img->data[c_idx];
					//printf("%d th pixel : ( %d %d %d ) \n", c_idx, img->data[c_idx].r, img->data[c_idx].g, img->data[c_idx].b);
					//printf("%d th pixel\n", c_idx);

					sum_r += pix.r;
					sum_g += pix.g;
					sum_b += pix.b;
					cnt_per_cell++;
					//#pragma omp atomic
					//cnt_per_cell++;
				}
			}//finish summation of each mosaic cell

			 //summation of all pixels in the img
			 //printf("Thread number : %d\n", omp_get_thread_num());
			 //#pragma omp atomic

#pragma omp critical
			{
				s_r += sum_r;
				s_g += sum_g;
				s_b += sum_b;
			}

			avg_r = sum_r / cnt_per_cell;
			avg_g = sum_g / cnt_per_cell;
			avg_b = sum_b / cnt_per_cell;

			//#pragma omp for collapse(2)
			//#pragma omp for private(k, l, c_idx)
			//#pragma omp for
			for (k = 0; k < c; k++) {
				for (l = 0; l < c; l++) {

					//printf("Thread number : %d\n", omp_get_thread_num());
					//current index
					c_idx = f_idx + img->width * k + l;
					//Pixel pix = img->data[c_idx];
					//printf("%d th pixel : ( %d %d %d ) \n", c_idx, pix.r, pix.g, pix.b);

					img->data[c_idx].r = avg_r;
					img->data[c_idx].g = avg_g;
					img->data[c_idx].b = avg_b;
					//check
					//printf("%d th pixel : ( %d %d %d ) \n", c_idx, img->data[c_idx].r, img->data[c_idx].g, img->data[c_idx].b);
				}
				//}
			}
			//#pragma omp end parallel 
			//finish summation of each mosaic cell
			//printf("\n");
		}
	}

	//the avg pixel values of the img
	int pixels = img->width*img->height;
	img->avg_r = s_r / pixels;
	img->avg_g = s_g / pixels;
	img->avg_b = s_b / pixels;
	//printf("%d %d %d", img->avg_r, img->avg_g, img->avg_b);

	return img;
}
void writeFile(const char *outputname, image* img) {

	//Before wring output file, debugging codes and check the metadata of image
	//printf("[Before writhFile()]\n");
	//printf("type : %s\n", img->type);
	//printf("size : %d * %d\n", img->width, img->height);

	FILE *fp;
	if (strcmp(img->type, "P3") == 0) { //P3 is plainText
										//printf("\n\nP3 in writeFile()\n\n");
										//open plain text file for writing
		fp = fopen(outputname, "w");
		//img format
		fprintf(fp, "%s\n", img->type);
		//img size
		fprintf(fp, "%d\n%d\n", img->width, img->height);
		// rgb component depth
		fprintf(fp, "%d\n", RGB_COLOR);
		// pixel data
		// fprintf() : write plainText file
		int i;
		for (i = 0; i < img->height*img->width; i++) {
			if (fp) {
				fprintf(fp, "%d %d %d ", img->data[i].r, img->data[i].g, img->data[i].b);
				//printf("%d pixel : (%d %d %d)\n", c_idx, img->data[i + j].r, img->data[i + j].g, img->data[i + j].b);
			}
			if (i%img->width == 0) fprintf(fp, "\n");
		}
	}
	else { //P6 is binaryText
		   //printf("P6 in writeFile()\n");
		   //open binary file for writing
		fp = fopen(outputname, "wb");
		//img format
		fprintf(fp, "%s\n", img->type);
		//img size
		fprintf(fp, "%d\n", img->width);
		fprintf(fp, "%d\n", img->height);
		// rgb component depth
		fprintf(fp, "%d\n", RGB_COLOR);
		// pixel data
		// fwrite() : write binary file
		fwrite(img->data, 3 * img->width, img->height, fp);
	}
	fclose(fp);
}

int main(int argc, char *argv[]) {
	//print_help();
	//return 1;
	if (process_command_line(argc, argv) == FAILURE) {
		return 1;
	}

	int c = atoi(argv[1]);
	char* mode = argv[2];
	//check mode
	if (strcmp(mode, "CPU") == 0) execution_mode = CPU;
	else if (strcmp(mode, "OPENMP") == 0) execution_mode = OPENMP;
	else if (strcmp(mode, "CUDA") == 0) execution_mode = CUDA;
	else execution_mode = ALL;

	char* inputname = argv[4];
	char* outputname = argv[6];
	//int i;
	//for (i = 1; i< argc; i++) printf("\narg%d = %s", i, argv[i]);
	//printf("[Input file] %s\n", inputname);
	//printf("[Output file] %s\n", outputname);
	//TODO: read input img file (either binary or plain text PPM) - DONE

	//check c
	//c should be the power of any 2
	if (c&(c - 1) != 0) {
		fprintf(stderr, "c should be the power of any 2.");
		//c should be less than width and height
		if (c > img->width || c > img->height) {
			fprintf(stderr, "c should be less than width and height of the image.");
			exit(1);
		}
	}

	img = readFile(inputname);

	//TODO: execute the mosaic filter based on the mode
	switch (execution_mode) {
	case (CPU): {
		//TODO: starting timing here - DONE
		double start, end;
		start = omp_get_wtime();
		//TODO: do_mosaic() 
		//TODO: calculate the average colour value in do_mosaic()
		img = do_mosaic(img, c);
		// Output the average colour value for the img
		printf("CPU Average img colour red = %d, green = %d, blue = %d \n", img->avg_r, img->avg_g, img->avg_b);
		//TODO: end timing here - DONE
		end = omp_get_wtime();
		double exe_time = (double)(end - start);
		printf("CPU mode execution time took %f ms\n", exe_time * 1000);
		break;
	}
	case (OPENMP): {
		//TODO: starting timing here - DONE
		double start, end;
		start = omp_get_wtime();
		//TODO: calculate the average colour value
		img = do_mosaic_omp(img, c);
		// Output the average colour value for the img
		printf("OPENMP Average img colour red = %d, green = %d, blue = %d \n", img->avg_r, img->avg_g, img->avg_b);

		//TODO: end timing here - DONE
		end = omp_get_wtime();
		double exe_time = (double)(end - start);
		printf("OPENMP mode execution time took  %f ms\n", exe_time * 1000);
		break;
	}
	case (CUDA): {
		printf("CUDA Implementation not required for assignment part 1\n");
		break;
	}
	case (ALL): { //run both modes CPU and OPENMP

		double start_cpu, end_cpu, start_openmp, end_openmp;

		//CPU
		start_cpu = omp_get_wtime();
		img = do_mosaic(img, c);
		printf("CPU Average img colour red = %d, green = %d, blue = %d \n", img->avg_r, img->avg_g, img->avg_b);
		end_cpu = omp_get_wtime();
		double exe_time = (double)(end_cpu - start_cpu);
		printf("CPU mode execution time took %f ms\n", exe_time * 1000);

		//OPENMP
		img = 0;
		img = readFile(inputname);
		start_openmp = omp_get_wtime();
		img = do_mosaic_omp(img, c);
		printf("OPENMP Average img colour red = %d, green = %d, blue = %d \n", img->avg_r, img->avg_g, img->avg_b);
		end_openmp = omp_get_wtime();
		exe_time = (double)(end_openmp - start_openmp);
		printf("OPENMP mode execution time took  %f ms\n", exe_time * 1000);

		break;
	}
	}

	//check -f option for designating output file format(P3/P6)
	if (argv[7] && argv[8]) {
		if (strcmp(argv[8], "PPM_BINARY") == 0 && strcmp(img->type, "P3") == 0) strcpy(img->type, "P6"); //P6 : PPM_BINARY
		else if (strcmp(argv[8], "PPM_PLAIN_TEXT") == 0 && strcmp(img->type, "P6") == 0) strcpy(img->type, "P3"); //P3 : PPM_PLAIN_TEXT
	}
	//save the output img file (from last executed mode)
	writeFile(outputname, img);

	//Not to close the cmd for debugging
	//system("pause");
	return 0;
}
void print_help() {
	printf("mosaic_%s C M -i input_file -o output_file [options]\n", USER_NAME);

	printf("where:\n");
	printf("\tC              Is the mosaic cell size which should be any positive\n"
		"\t               power of 2 number \n");
	printf("\tM              Is the mode with a value of either CPU, OPENMP, CUDA or\n"
		"\t               ALL. The mode specifies which version of the simulation\n"
		"\t               code should execute. ALL should execute each mode in\n"
		"\t               turn.\n");
	printf("\t-i input_file  Specifies an input img file\n");
	printf("\t-o output_file Specifies an output img file which will be used\n"
		"\t               to write the mosaic img\n");
	printf("[options]:\n");
	printf("\t-f ppm_format  PPM img output format either PPM_BINARY (default) or \n"
		"\t               PPM_PLAIN_TEXT\n ");
}
int process_command_line(int argc, char *argv[]) {
	if (argc < 7) {
		fprintf(stderr, "Error: Missing program arguments. Correct usage is...\n");
		print_help();
		return FAILURE;
	}

	//first argument is always the executable name

	//read in the non optional command line arguments
	c = (unsigned int)atoi(argv[1]);

	//TODO: read in the mode
	char* mode = argv[2];
	//TODO: read in the input img name
	char* input_img = argv[4];
	//TODO: read in the output img name
	char* output_img = argv[6];
	//TODO: read in any optional part 3 arguments
	char* options = argv[7];
	/*
	printf("mode : %s, input_img = %s, output_img = %s, options = %s",
	mode, input_img, output_img, options);
	*/
	return SUCCESS;
}


