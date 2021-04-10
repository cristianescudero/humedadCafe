#include "Imaginarios.hpp"

Imaginarios::Imaginarios(float op){
    cout<<"Por favor ingresar los numeros"<<endl;
}

void Imaginarios::IngresarInfo1(){

    float _real1, _ima1, _real2, _ima2;
    cout<<endl;
    cout<<"Ingresar el valor real"<<endl;
    cin >>_real1;
    cout<<"Ingresar el valor imaginario"<<endl;
    cin >>_ima1;
    cout<<endl;

    real1 = _real1;
    ima1 = _ima1;
}

void Imaginarios::IngresarInfo3(){

    float _mag, _ang;
    cout<<"Ingresar la magnitud"<<endl;
    cin >>_mag;
    cout<<"Ingresar el angulo"<<endl;
    cin >>_ang;

    mag = _mag;
    ang = _ang;
}


void Imaginarios::IngresarInfo2(){

    float _real1, _ima1, _real2, _ima2;
    cout<<endl;
    cout<<"Primer numero"<<endl;
    cout<<"Ingresar el valor real"<<endl;
    cin >>_real1;
    cout<<"Ingresar el valor imaginario"<<endl;
    cin >>_ima1;
    cout<<endl;
    cout<<"Segundo numero"<<endl;
    cout<<endl;
    cout<<"Ingresar el valor real"<<endl;
    cin >>_real2;
    cout<<"Ingresar el valor imaginario"<<endl;
    cin >>_ima2;

    real1 = _real1;
    ima1 = _ima1;
    real2 = _real2;
    ima2 = _ima2;

}
void Imaginarios::SumaIma(){

    float sum1 = real1 + real2;
    float sum2 = ima1 + ima2;

    if(sum2<0){
        cout<<"El resultado de la suma es: "<<sum1<<sum2<<"i"<<endl;
    }
    else {
        cout<<"El resultado de la suma es: "<<sum1<<"+"<<sum2<<"i"<<endl;
    }
}

void Imaginarios::RestaIma(){

    float sum1 = real1 - real2;
    float sum2 = ima1 - ima2;

    if(sum2<0){
        cout<<"El resultado de la resta es: "<<sum1<<sum2<<"i"<<endl;
    }
    else {
        cout<<"El resultado de la resta es: "<<sum1<<"+"<<sum2<<"i"<<endl;
    }
}

void Imaginarios::MultiIma(){

    float mult1 = ((real1*real2)-(ima1*ima2));
    float mult2 = ((real1*ima2)+(ima1*real2));

    if(mult2<0){
        cout<<"El resultado de la resta es: "<<mult1<<mult2<<"i"<<endl;
    }
    else {
        cout<<"El resultado de la resta es: "<<mult1<<"+"<<mult2<<"i"<<endl;
    }

}

void Imaginarios::DivIma(){

    float div1 = ((real1*real2)+(ima1*ima2))/((pow(real2,2))+(pow(ima2,2)));
    float div2 = ((ima1*real2)-(real1*ima2))/((pow(real2,2))+(pow(ima2,2)));

    if(div2<0){
        cout<<"El resultado de la resta es: "<<div1<<div2<<"i"<<endl;
    }
    else {
        cout<<"El resultado de la resta es: "<<div1<<"+"<<div2<<"i"<<endl;
    }

}

void Imaginarios::ConvRecPol(){

    float magnitud = sqrt((pow(real1,2))+(pow(ima1,2)));
    float angulo = atan (ima1/real1) * 57.29578;
    cout<<"El resultado de la conversion es: "<<magnitud<<"|___"<<angulo<<endl;

}
void Imaginarios::ConvPolRec(){

    float ValorReal = mag * cos(ang*(3.1416/180.0));
    float ValorImag = mag * sin(ang*(3.1416/180.0));

    cout<<mag<<" y "<<ang<<endl;

    if(ValorImag>0){
        cout<<"El resultado de la conversion es: "<<ValorReal<<"+"<<ValorImag<<"i"<<endl;
    }
    else {
        cout<<"El resultado de la conversion es: "<<ValorReal<<ValorImag<<"i"<<endl;
    }

}
