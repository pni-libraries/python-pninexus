
import pni.nx.h5 as nx

#define FNAME "d4_test.h5"
#define SAMPLE "JJ0815"

#include "NXHelper.hpp"


class D4File:
	NXFile _file;
	NXGroup _instrument;
	NXGroup _sample;
	NXGroup _entry;
	String _sample_name;

	Record _record;

protected:
	void _setup_storage_ring();
	void _setup_bending_magnet();

public:
	D4File(const String &fname);
	virtual ~D4File();

	virtual void close();
	virtual void set_sample(const String &sample_name);
	virtual void setup_scan(const String &scan_name);

	NXField &operator[](const String &key);
};

D4File::D4File(const String &fname){
    _file = NXFile::create_file(fname,true,0);
}

D4File::~D4File(){
	close();
}

void D4File::close(){
	_file.close();
	_record.clear();
}

void D4File::set_sample(const String &sample_name){
	_sample_name =sample_name;
}

NXField &D4File::operator[](const String &key){
	return _record[key];
}

void D4File::_setup_storage_ring(){
	NXGroup doris;
	Float32Scalar emittance_x(410,"emittance_x","nm-rad","emittance in horizontal direction");
	Float32Scalar emittance_y(12,"emittance_y","nm-rad","emittance in vertical direction");
	Float32Scalar energy(4.45,"energy","GeV","positron energy");
	Float32Scalar current("current","mA","ring current");


	doris = _instrument.create_group("DORIS","NXsource");
    NXField f = doris.create_field<Float32>("distance");
    f.attr<String>("unit").write(String("m"));
    f.attr<String>("description").write(String("distance to DORIS"));
    f.write(Float32Scalar(40.,"distance","m","distance to DORIS"));

	doris.create_field<String>("name").write(String("DORIS"));
	doris.create_field<String>("type").write(String("Synchrotron X-ray Source"));
	doris.create_field<String>("probe").write(String("positron"));
    f = doris.create_field<Float32>("emittance_x");
    f.write(emittance_x);
    f.attr<String>("unit").write(emittance_x.unit());
    f.attr<String>("description").write(emittance_x.description());

	f = doris.create_field<Float32>(emittance_y.name());
    f.write(emittance_y);
    f.attr<String>("unit").write(emittance_y.unit());
    f.attr<String>("description").write(emittance_y.description());

	f = doris.create_field<Float32>(energy.name());
    f.write(energy);
    f.attr<String>("unit").write(energy.unit());
    f.attr<String>("description").write(energy.description());

	//doris current might be written during scan
	//_record["doris_current"] = doris.createNumericField(current);
}

void D4File::_setup_bending_magnet(){

}

void D4File::setup_scan(const String &scan_name){
	NXGroup g;

	_entry = _file.create_group(scan_name,"NXentry");
	_instrument = _entry.create_group("instrument","NXinstrument");
	_sample = _entry.create_group("sample","NXsample");
    _entry.create_group("control","NXmonitor");


	//setup the basic system
	_setup_storage_ring();
	_setup_bending_magnet();

    //create all the slits available at the beamline
    NXHelper::createNXslit(_instrument,"Slit1","mm","slit after bending magnet");
    g = NXHelper::createNXslit(_instrument,"Slit2","mm","slit in front of sample");
    //slit 2 resides on a tower with a tilt
    NXHelper::createNXpositioner(g,"R1","mm","R1","angle first flight tube");
    NXHelper::createNXpositioner(g,"Z1","mm","Z1","height first flight tube");

    NXHelper::createNXslit(_instrument,"Slit3","mm","slit after sample");
    NXHelper::createNXslit(_instrument,"Slit4","mm","slit in front of detector");

    //right after the first slit we have a mirror 
    //that must be taken into account


    //add motors to sample
    NXHelper::createNXpositioner(_sample,"XS","mm","XS","sample x-translation");
    NXHelper::createNXpositioner(_sample,"YS","mm","YS","sample y-translation");
    NXHelper::createNXpositioner(_sample,"GUS","mm","GUS","upper cradle");
    NXHelper::createNXpositioner(_sample,"GLS","mm","GLS","lower cradle");
    NXHelper::createNXpositioner(_sample,"OMA","mm","OMA","sample rotation");
    NXHelper::createNXpositioner(_sample,"OMS","degree","OMS","angle of incidence");
    

}

int main(int argc,char **argv){
	D4File file(FNAME);

	file.set_sample(SAMPLE);
	file.setup_scan("scan_1");

	//run experiment
	for(UInt64 i=0;i<100;i++){
		//file["doris_current"]<<Float32Scalar(140-i*0.1,"current","mA");
	}


}




